# Writing documentation that teaches

TamerOp's documentation should help readers understand why finite encoding
is the central object and how to use it to answer mathematical questions.
A reader should be able to follow the argument, interpret a result, and
choose a sensible next step.

Two principles govern documentation contributions: develop a connected
finite-encoding learning narrative, and teach in approachable language.
Lessons develop that narrative; independently consulted guides, reference
entries and reports begin with their own reader question and link the context
needed to answer it. Every family should explain its objects and claims clearly.

The [first learning-path brief](learning_path.md) records the initial page
sequence, reader questions, and expected mathematical answers. The
[API inventory guide](api_inventory.md) explains how to maintain the generated
binding inventory and the explicit reference-writing backlog.

Benchmark reports also follow the shared [visual standard](benchmark_style.md),
using QPA as the aesthetic reference. Its page, table and figure conventions
complement the narrative guidance here without requiring identical plot types
for different studies.

## Choose an article's purpose before its format

Use the [article inventory](article_inventory.md) for existing and planned
treatments. Assign one primary editorial family and a clear reader question.
Topic membership, learning-route membership and publication format are separate
choices. A notebook, its generated website lesson and its executed download
have one canonical authored source and one article identity.

| Family | Reader's question | What the article should accomplish |
| --- | --- | --- |
| Mathematical chapter or worked lesson | What does this mean, and why does it matter? | Develop understanding through definitions, examples, computations and interpreted figures. Allow both example-first and definitions-first entrances. |
| Library guide | What can TamerOp do here, and how do I use its capabilities together? | Develop a workflow, explain returned objects and meaningful choices, then explore a few useful variations. |
| Task recipe | How do I accomplish this particular task? | Give a focused procedure, essential conditions and a recognizable result. Installation and operational support belong here. |
| API reference | What exactly does this operation accept, return and guarantee? | Specify arguments, defaults, domains, shapes, boundary behavior, errors and mathematical scope, with small examples. |
| Implementation account | How is this computed, and why this design? | Explain representations, execution choices, correctness conditions, reuse and costs, with sources credited where used. |
| Benchmark report | What does the measured evidence establish? | Explain the comparison question, matched outputs, conditions, results and limitations. |
| Contributor or maintainer guide | How do I change, document, test, benchmark or release this project consistently? | Give procedures and standards for project work without making them prerequisites for library users. |

The glossary, bibliography, maps, data dictionaries, changelog and planning
records are supporting resources. Inventory them without forcing them into
a narrative genre. A gallery indexes figures and their canonical treatments;
a worked application takes its family from the question it teaches or solves.

Library guides retain the approachable style of the lessons, with the library's
capabilities as their subject. Start with a meaningful public workflow and
interpret its result before introducing additional choices. Figures should
show actual results, comparisons or the consequences of a choice. Reveal
options progressively, and keep styling and export details skippable. Link
the mathematical treatment and precise API contract at the point of need.
Do not turn the guide into a keyword catalog or repeat the introductory story.

A topic need not receive an article in every family. A short recipe can be a
section of a guide; split it when independent lookup benefits the reader.
Split an existing article when its purpose changes substantially, not whenever
it contains mathematics, code or implementation context. Keep necessary
assumptions beside the operation even when a longer explanation lives elsewhere.

### Move from understanding to exploration to a defined task

**Mathematics → Using TamerOp → Task recipes** describes a spectrum from
understanding an idea, through exploring the library's capabilities and choices,
to accomplishing a particular task. It is a distinction in reader purpose,
not a prerequisite chain or a ranking of difficulty. Readers can enter any of
these collections directly. A demanding algebraic recipe can be more advanced
than an introductory mathematical lesson.

A lesson might show why dimensions do not determine a persistence module.
A library guide explores the spaces, maps and coordinate views available for
an encoded object, helping readers decide what to inspect. A recipe starts with
two supplied parameters and ends with their structure-map matrix and its rank.
Give each treatment enough local context to be useful and link its neighbors
where their explanation becomes relevant. The introduction should make these
three purposes visible without presenting them as a course everyone must finish.

### Write recipes around an attainable result

A recipe serves someone who already knows what they want to do. Use a concrete,
task-oriented title, identify the starting object or data and intended result,
then give a runnable procedure. Make success recognizable through an expected
value, a displayed result, a saved artifact or a check that the reader can
interpret. Prefer a small self-contained input; when a prepared object is
necessary, link precisely to its construction and state what it must contain.

Let code carry the procedure, with concise explanations of consequential choices
and outputs. Keep necessary mathematical hypotheses beside the step that uses
them. A structure-map recipe, for example, must distinguish a zero map between
comparable parameters from the absence of a prescribed map between incomparable
ones. State required packages and execution context; avoid hidden notebook state,
unexplained helper wrappers and an exhaustive option catalog.

Use a figure when it helps the reader inspect the result, rather than requiring
one for every task. Link to a guide for choosing among capabilities, to theory
for the underlying argument, and to reference for exact contracts at the point
of need. Include a small variation or recovery note when it resolves a likely
obstacle; broader exploration belongs in the guide. These responsibilities do
not prescribe a fixed section template, code-to-prose ratio or page length.

Installation remains a recipe with a permanent upper-sidebar entrance. Existing
recipe classifications record an editorial home, not proof that the page
already follows this style. Use inventory migration notes for unfinished
adaptations and planned records for independently useful procedures. Do not
create a page for every keyword or mirror all guides with shortened copies.

### Continue the mathematics after encoding

Mathematical lessons cover both construction of a finite encoding and the
questions it makes possible. They include algebraic operations, categorical
transport, invariants, summaries, vectorization and the interpretation of
figures. An advanced topic or an executable computation does not by itself
turn a lesson into a library guide. The lesson teaches the operation's meaning
and consequences; a library guide teaches how to choose and combine the
available software workflows.

Treat the retained encoding as a junction for several mathematical branches.
Readers can investigate ranks, slices and features without first completing
resolutions or derived algebra. Reuse a recognizable object where it teaches
the next idea, and introduce a small new example when necessary. State the
base category, hypotheses and retained information at each transition; ambient
recovery does not establish ambient invariance of finite-category Ext or Tor.

Use visuals throughout, and allow a later interpretation lesson to bring views
together. A dimension heatmap, a signed summary, an interval approximation and
a module decomposition assert different things. Explain the transformation
from object to summary to numerical feature or displayed image; identify what
is preserved and what is lost. A stable feature claim needs its stated metric
and hypotheses, not merely a smooth-looking picture.

The [documentation programme](documentation_plan.md) plans the complete set of
branches; the [learning brief](learning_path.md#after-the-first-encoding-the-mathematical-curriculum)
develops the opening examples. Add cards to the reader-facing map when the
treatments exist, with routes that expose choices rather than turning the
inventory into a mandatory reading list.

Keep direct finite diagrams, lattice presentations and geometric region
presentations visible alongside filtered points, images, graphs and complexes.
Use algebraic and categorical operations as substantive investigations with
their own examples. They should not appear only as advanced appendices to a
filtration-to-invariant pipeline. The shared encoding joins these entrances;
supported direct ordinary-persistence computations retain their honest shorter
route.

### Use boundary examples to prevent duplication

For structure maps, different treatments answer different questions:

| Family | Boundary example |
| --- | --- |
| Lesson | Why dimensions alone do not determine a persistence module. |
| Library guide | How to explore the spaces and maps of a constructed object and choose what to inspect. |
| Recipe | How to recover a map at two supplied comparable parameters. |
| API reference | Which inputs are labels or parameters, when they are comparable, and the returned matrix's shape and field. |
| Implementation account | How map storage, composition, coordinate recovery and caching work. |
| Benchmark report | What a defined measurement establishes about map construction, queries or reuse. |
| Contributor guide | How to extend or validate the map-query subsystem. |

Coefficient choice is another useful test: theory explains field dependence;
a guide helps choose exact or numerical computation; reference gives supported
fields and tolerance semantics; implementation explains arithmetic algorithms
and actual routing. These examples define ownership of explanations, not a
requirement to write every possible treatment. Prefer one full explanation
with contextual links and brief reminders over several accounts that drift.

### Give optimization a stable explanatory home

A faster computation normally updates an existing implementation account.
Explain the representation, algorithm, output-sensitive work, materialization,
reuse and remaining costs there. A usage guide owns choices the reader can
make; reference owns their observable contracts. Mathematics changes when the
represented object, hypotheses or guarantees change. Measured timings and
ratios belong to a versioned benchmark report under its declared protocol.
Link those accounts instead of copying measurements into several pages or
creating a mathematical lesson for every kernel optimization.

For lazy or cached work, distinguish first use, compiled computation with
uncached mathematics, declared shared preparation and retained-result queries.
Explain what is retained and what is recomputed. Memory and startup costs have
their own meaning; a cheap query does not establish a cheap complete workflow.

## Keep collections, topics and page outlines distinct

The upper sidebar always offers **Introduction**, **Installation**, the
**Learning map** and the **Topic map**. Beneath those entrances, six collections
organize the usable articles by purpose:

| Collection | What readers find |
| --- | --- |
| Mathematics | Mathematical chapters and worked lessons, including the questions made possible by a finite encoding. |
| Using TamerOp | Library guides that explore capabilities, returned objects and consequential choices. |
| Task recipes | Focused procedures for accomplishing a defined task and recognizing its result. |
| API reference | Precise operation and shared-contract reference. |
| Implementation | Accounts of how coherent computations work. |
| Benchmarks | Completed comparisons and their evidence. |

Only the current collection's branch starts open. Readers can open another
branch deliberately; keep the hierarchy shallow and add a grouped landing
page when a collection grows. Contributor guidance is reached through the
persistent sidebar footer link. Link a relevant testing or authoring guide within an
article when the reader needs it, without making contributor procedures part
of the main learning sequence.

Installation stays permanently above the collections even though its editorial
family is `recipe`. Its upper link and recipe listing resolve to the same
canonical article. The six collections express purpose; their display order
does not impose a reading sequence.

The local page outline contains that article's H2 and H3 headings. Keep it
separate from the collection hierarchy: a section heading is not another
article. Search identifies article type and topic so that a lesson, a usage
guide and an algorithm account with similar titles remain distinguishable.

The site derives collection and topic membership from `article_inventory.toml`
and published destinations from `publication.toml` and authored site pages.
`navigation.toml` supplies display grouping and order; it must not become a
second list of articles. Public collections show authored, usable treatments.
Keep proposals and migration notes in the inventory, and retain one canonical
source even when an article appears through several topic links. The
[site maintenance guide](README.md#maintain-site-navigation) explains the build
and review responsibilities.

Simplifying a sidebar should not remove a published subject index. Keep useful
subject landings reachable through the collection even when a shorter article
list no longer needs grouped navigation. If their articles move to a different
editorial family, make that distinction clear in the links. Each index continues
to point to the canonical articles; there is no need to duplicate their content.

## Separate learning routes from topic exploration

Maintain two complementary navigation views. The reading map answers which
direction a reader can take to understand the mathematics. A topic map answers
where the documents related to an area can be found. Article families determine
editorial purpose; topic clusters organize subjects, not levels of difficulty.

The topic map is a mind map of mathematical and computational areas. Each
cluster gathers available lessons, library guides, API entries, implementation
accounts and reports about that area, with restrained text labels for type.
Use short titles and reader questions to explain what each link offers. An area
need not contain every kind of article.

- Start with a small cluster overview and let the reader open or enter a topic
  to see its documents. Add space or a topic landing view before shrinking text
  or drawing a dense web of every cross-reference.
- Use enclosures or undirected branches for membership. Related-topic links
  express relevance, not prerequisites. Do not borrow the reading map's
  directional continuation arrows for these relationships.
- Keep one inventory identity and canonical source for an article belonging to
  several topics. Cross-topic shortcuts refer to that record; do not maintain
  separate summaries or copies in each cluster.
- Derive membership from the article inventory and resolve destinations from
  publication metadata or clearly labeled repository sources. Public maps
  include accessible authored material; planned treatments stay in the
  inventory. Source existence alone does not establish publication readiness.
- Provide a linked outline with the same destinations, keyboard navigation,
  visible focus and readable narrow-screen and light/dark layouts. Type and
  meaning must not depend on color or an interactive renderer.
- Keep the two maps separate and mutually discoverable. A topic cluster can
  reveal practical and deeper accounts without making them required stops on
  the mathematical route.

The generated topic map and topic landing pages use this same article metadata.
Their links expose published treatments across the collections. The overview,
linked topic listings and search must agree without another independently
authored article list. The learning map keeps its own suggested progression;
topic membership never creates a learning arrow or changes a lesson's footer.

## Maintain the article inventory during gradual changes

Record the intended type, reader question, topics, canonical source and next
editorial action before writing or restructuring a treatment. Keep individual
migration decisions in the inventory and enduring principles in this guide.
An existing page can remain useful while awaiting adaptation; its age or
directory does not establish that it should be removed.

Revise one coherent area at a time. Preserve accurate explanations and essential
assumptions, split or merge only where purpose warrants it, and update incoming
links when moving content. Keep one authoritative source. Review nearby pages
for duplicated explanations and conflicting terminology. Record completion and
validation in contributor records, not in the lessons themselves.

Article metadata does not replace the specialized manifests: learning routes
belong to `reading_map.toml`, build routes to `publication.toml`, API-family
coverage to `api_coverage.toml`, and observed symbols to the generated runtime
inventory. `navigation.toml` owns display grouping only. Preserve those
responsibilities and link related records.

## Connect lessons along the reading route

The [reader-facing map](src/reading_map.md) makes these routes visible. When
adding a lesson to a reading route, record its question and continuations in
`reading_map.toml`, using its canonical source path. Keep the graph acyclic,
and distinguish suggested routes from mandatory prerequisites. The site uses
one graph for the linked diagram and text outline, and labels repository
destinations separately. Add unwritten lessons to the backlog; add their
cards when the treatments exist. See the [map maintenance guide](README.md#maintain-the-reading-map)
for layout and browser checks.

End a lesson on a declared route with a motivated next question and one primary
continuation.
Offer at most one alternative when it serves a different reader's purpose.
Record these destinations as `next` and, if needed, `alternate` in the map;
the final section (or final notebook Markdown cell) links to them in that order.
The build checks those links and uses the same primary destination in the site
footer. Sidebar order is a catalog, not an instruction to read every page.
Use the reading map for the wider choices instead of repeating the curriculum
at the end of each lesson. Supporting references need no artificial sequence.

## Preserve choices as the reading map grows

The map helps a reader choose a useful next question. Its visual arrangement
must communicate that choice without requiring a paragraph to correct a
misleading impression.

The target is complete mathematical coverage through several views of one
graph: every published lesson appears in at least one focused diagram and in
the shared accessible outline. This includes advanced definition-led chapters.
The overview is selective, not the underlying collection. Keep minimum
knowledge, suggested continuations and thematic cross-links distinct. The
[full programme](documentation_plan.md#one-graph-several-views-complete-mathematical-coverage)
specifies this expansion; it requires extending the current renderer and route
model as the branches are authored, not putting unwritten pages into live maps.
Preserve these principles when adding or moving pages:

- **Make alternative starts equally visible.** Keep the example-first and
  definitions-first routes at the same visual level, with comparable card size
  and emphasis. Label the purpose of each start. A page remains a starting
  point even when an optional arrow also leads into it; incoming arrows do not
  determine entry status. Add another start only for a distinct reader need.
- **Give arrows a consistent meaning.** Solid arrows suggest continuation;
  labeled, dashed arrows offer optional detours. The branch into the full
  definitions must leave the direct route visible. Use words and line styles
  together, so color is never needed to distinguish them. Converging arrows
  offer alternative ways to arrive, not a requirement to finish every branch.
  State genuine prerequisites in the lessons where readers need them.
- **Keep the shared mathematical destination clear.** Both introductory routes
  reach finite encodings. Later branches should answer different questions
  about the retained object, using the continuing examples where useful.
  Readers should see why a branch exists before following it.
- **Separate learning from supporting tasks.** Place installation beside the
  notebook as help for running it; saved figures remain readable without it.
  Keep API conventions and other reference material in the grouped guides or
  nearby prose. A new documented capability does not automatically need a
  prominent card or another arrow in the main diagram.
- **Spend visual space on decisions.** Give each card one recognizable topic
  and one reader question. Favor generous spacing, short labels, restrained
  color, and routes that can be traced without crossing cards or labels.
  Keep the diagram selective rather than drawing every cross-reference.
  If a branch grows crowded, group its supporting material or give it a
  focused linked map before shrinking text or packing in more connectors.
- **Preserve the same choices for every reader.** On narrow screens, show both
  starts before horizontal scrolling. Retain a linked text outline, keyboard
  access, visible focus, and legible light/dark layouts. Essential navigation
  must work without a JavaScript diagram renderer. The outline and diagram
  must agree about entry points, detours, and continuations.
- **Keep one source of navigational truth.** Maintain canonical destinations,
  entry labels, and route distinctions in `reading_map.toml`. Keep lesson
  endings, the outline, and generated footer links in agreement. The sidebar
  is a catalog; it must not impose a competing reading sequence. Unwritten
  lessons belong in the authoring backlog rather than linked map cards.

Preserve these relationships rather than freezing the current coordinates,
number of cards, or exact layout. Review the rendered map as a newcomer:
where can I start, which detours can I skip, and where does my choice lead?
Automated checks support this review; they cannot establish that the visual
hierarchy teaches those choices. Use the
[map expansion checklist](README.md#map-expansion-checklist) for each addition.

## Keep lessons about the reader's task

Describe the mathematics and the available workflow directly. Lessons do not
need implementation status, release chronology, dated validation reports,
assertion counts, or general feature roadmaps. Keep development plans in the
backlog and verification evidence in contributor or testing material. This
applies to current status claims as well as stale ones.

A brief **figure placeholder** is an exception while the intended visual is
missing. Put it where the figure will support the argument, label it clearly,
and describe the mathematical objects and the fact the reader should notice.
Avoid implementation dates, progress reports, or claims that its interaction
already works. Keep the surrounding explanation understandable without it;
replace the placeholder with the figure and an interpretive caption when ready.

Keep qualifications that affect interpretation or use: the supported inputs,
mathematical assumptions, approximation or censoring, and requirements such as
a live Julia session. Distinguish a mathematical schematic from a computed
plot. Omitting development status must never imply that an unsupported operation
exists or that an unproved claim holds.

## Explain implementation rather than its development history

Implementation references answer how an algorithm works and why its choices
make sense. The [exact rational coordinates account](implementation/qq_coordinates.md)
is a model for their depth and explanatory style. Preserve its principles;
adapt the section order, examples and amount of detail to the operation.
There is no required chapter outline, figure count or code quota.

These accounts form their own category, outside the reading map and supporting
usage guides. A reader may arrive directly with a question about an algorithm.
Briefly establish its mathematical purpose and prerequisites, linking to the
relevant theory or usage page without repeating that treatment. Scope each
account around a coherent computation or implementation decision; a source
file or API family need not correspond to one chapter.

### Follow the operation from mathematics to execution

Begin with the question the computation answers and why it occurs in the
library. State the input, returned mathematical object, assumptions and choices
that affect the answer. Introduce dimensions, notation and basis conventions
where needed. Then explain the central mechanism before expanding into
specializations. Follow the dependencies of the argument rather than the order
of functions in a source file.

Make the connection between equations and stored data explicit: what is
constructed, retained, recomputed and returned? Explain why a shortcut preserves
the required result. Keep runtime correctness checks and their mathematical
justification: they are part of the algorithm. Separate a candidate from a
certificate, an exact guarantee from a heuristic, and a default check from a
precondition that the caller must satisfy when that check is disabled.

Choose small worked examples that expose a consequential design choice. In
the coordinate account, dependent leading rows explain why row selection needs
a fallback; changing an unselected entry of the right-hand side explains why
the candidate still needs a membership check. Carry the same example far enough
to explain the mechanism.
A second example is useful when it reveals a different issue, not merely to
increase coverage. Say when a small illustration does not activate the actual
optimization's size threshold.

Use short executable examples when they connect the explanation to a real
entrypoint. Identify owner APIs and internal structures as such, explain any
explicit options needed to reach the illustrated path, and interpret the result.
Pseudocode can clarify a longer algorithm without reproducing its implementation.
Link shared machinery to its existing account instead of explaining it again
in every consumer's chapter.

### Explain choices, reuse and costs precisely

Whenever multiple methods appear, explain how execution reaches each one:
the default route, explicit choices, thresholds, retained-state overrides and
fallback conditions. Distinguish selection between requests from a fallback
within one request, and distinguish package-level routing from algorithms
inside a dependency. A list of available methods alone does not explain the
implementation. A compact decision table can make precedence easier to see.
Label configurable thresholds as defaults and distinguish empirical crossover
rules from mathematical restrictions.

Explain the resulting tradeoffs: which work is saved, which conversion or
temporary allocation is introduced, and which inputs benefit or incur extra
cost. Separate construction from repeated queries and retained storage from
temporary work. State ownership, invalidation and concurrency constraints when
they affect safe reuse. Distinguish a small internal payload from the complete
returned object. Give meaningful cost estimates with their assumptions; counts
of field operations or stored entries do not by themselves bound the cost of
large exact coefficients.

### Use figures and sources to support the explanation

Include a diagram when it clarifies data flow, reuse, a representation or a
decision that prose leaves difficult to follow. Keep mathematical labels
consistent with the equations, distinguish preparation from repeated work,
and show relevant return, rejection or fallback branches. Use readable labels,
useful alternative text and a caption that states what the reader should
notice. A matrix example or a short table may be more informative than a
diagram. Let each visual resolve a specific explanatory need.

Link source locations and useful symbols where they let readers follow the
mechanism; a compact source table near the end can support this without
interrupting the argument. Cite papers and software beside the ideas they
support, especially specialized algorithms, bounds and less familiar
constructions. Distinguish directly used software, implemented or adapted
methods, mathematical background, and related alternatives. Verify the
attribution against both source and implementation; similarity alone does not
establish historical inspiration or mean that a paper's whole method is used.
Maintain these citations in the shared [bibliography](implementation/references.md),
and end the account with a simple bibliography link.

### Keep the account about the algorithm

Keep benchmark campaigns, before/after timing tables, test-suite accounts,
review dates, source hashes and other audit bookkeeping in the separate
benchmarking or contributor material. When an otherwise arbitrary choice needs
context, a brief statement that it was chosen after benchmarking alternatives
is enough. Explain the resulting tradeoff rather than recounting the experiments.
Apply this distinction throughout the prose, including introductions, captions
and closing paragraphs, not just when choosing sections.

Supporting evidence remains necessary for authoring, but its collection and
review process are not the subject of the account. Avoid closing commentary
about the document's provenance or why its bibliography exists. Read the
finished explanation for continuity: can someone follow the operation, see
why its important choices work, identify which route their call takes, and
understand its costs and limitations? Use these questions to judge the page;
do not turn them into mandatory reader-facing sections.

## Follow the mathematical question

The [finite-encoding introduction](finite_encodings.md) gives the shared
progression. Begin with what ordinary persistence can describe. Explain what
changes with several parameters and why summaries leave questions unanswered.
Introduce finite encoding as a way to retain the module in a finite model,
then follow that object through construction, algebra, and chosen summaries.

Preserve the project's mathematical origin: TamerOp began as an implementation
of Ezra Miller's theory of modules over posets. Explain the connection between
tameness and finite descriptions when introducing the name or the architecture.
Separate the theory's general statements from the library's supported
constructions and their assumptions.

Each lesson on a reading route should answer a question that the earlier
explanation has made meaningful. Say what object we begin with, what we learn
or construct, and why the next question follows. Independently consulted
articles establish their own question and link relevant context without
requiring readers to traverse the curriculum. Carry a worked example across
related pages where it helps readers recognize the same object.

A reference entry can do this briefly: begin with the mathematical purpose,
then explain its arguments, returned object, and assumptions. A utility
description should explain its actual role. Do not imply that every direct
ordinary-persistence call constructs an encoding.

## Make the explanation approachable

Write for a thoughtful reader who may be new to either Julia or the topic.
Introduce only the terminology needed for the next idea, and explain that
terminology where it becomes useful.

For example, “the map assigning each original parameter a finite label”
gives a reader something concrete before the name *classifier* is introduced.
“Save the encoding” is enough for a first-use instruction; *serialization*
becomes useful when discussing the file format and what it preserves.

Explain every new mathematical symbol and what an equation says in words.
Around code, describe the input, the important options, the returned object,
and how to interpret the answer. Distinguish Julia commands from terminal
commands. Prefer the documented public operations and accessors so readers
can recognize the same steps in their own work.

Use short, connected paragraphs. Give a small example before a broad catalog.
Avoid using “obvious,” “trivial,” or “simply” in place of an explanation.
Advanced mathematics deserves the same care as introductory material.

## Preserve the mathematics

Approachable exposition still needs precise assumptions. Explain which
module and parameter domain are represented, where maps enter the argument,
and which choices affect the result. Distinguish a coefficient field from
the arithmetic used for geometric grades.

When explaining why an input is tame, identify the finite structure supplied
by its construction: for example, a filtration by subcomplexes of a fixed
finite complex or a finite fringe presentation. Show how it supplies an
encoding, including the maps. This lets readers rely on established
constructions without having to prove tameness anew. A finite file, a short
program, or a finite set of queries alone does not establish the condition
for an arbitrary module.

For closure under kernels, images, quotients, or homology, explain how the
maps or differentials fit the finite description. State the object, morphism,
and geometric hypotheses of an abelian-category theorem where they are used.
Keep the claim about the represented model distinct from any claim about an
underlying continuum object. The [practical tameness chapter](practical_tameness.md)
develops these distinctions through the continuing examples.

Do not infer encoding-independent Ext or Tor from recovery of an ambient
module or from abelian closure. State the finite category of a derived
computation and link to the [comparison hypotheses](math_categories.md) when
relevant.

Use examples with known answers. Verify maps as well as dimensions when
the claim concerns a module or a comparison. A successful computation supports
a worked example; a general theorem needs a reference or derivation.

Figures should advance the explanation. Reuse labels across a parameter
picture, its finite poset, and the corresponding spaces and maps. Explain in
the caption what the reader should notice. Include the source of generated
figures and the conditions under which the displayed result is computed.

## Plan the visible result of a notebook

Before writing executable cells, specify what the reader will see and what
they should conclude from it. For a substantial computational example, use this
sequence where the objects admit a useful picture:

1. Show the input and state a question the reader can answer by prediction.
2. Compute one result through the public API and display a compact mathematical
   answer: an interval, a matrix, a count, or another appropriate summary.
3. Place the corresponding figure immediately after that computation. Explain
   one visible fact and connect it to the answer just checked.
4. Change one meaningful input or parameter, predict its effect, and compare
   the results. Keep the original available rather than overwriting it.

For example, display the ring's input grades, its active top cells before and
after the center appears, and the interval `[0,5)`. A square-module lesson
instead follows a parameter into its finite label, stalk, and selected map.
Use the mathematical object appropriate to the lesson; a finite poset with
matrices need not have a source point cloud or a representative geometric cycle.
An exact matrix or small commutative diagram can be more useful than a heatmap.

Comparison figures should retain coordinate limits, orientation, units, color
scales and degree conventions when those quantities are comparable. State any
necessary change of scale. Label landscape axes by the actual parameter, and
image axes by the represented coordinates, rather than unexplained array
indices. A preview of the filtration field should use the values supplied to
the computation; identify any interpolation or display-only transformation.

Introduce a slider or linked inspector after the static example is understood.
Preserve a few informative states as static figures, so the argument remains
readable without a running kernel. The published page and downloadable executed
notebook must contain the required figures. A source notebook may have cleared
outputs, but its publication build must capture them; code containing
`display(...)` alone does not meet this requirement. Check the actual notebook
frontend separately from a headless execution or a standalone browser server.

Use existing package visualization operations for mathematical results. When a
reusable view is missing, record it in the visualization backlog or authoring
brief; a concise figure placeholder may mark its intended place in the lesson.
Label a schematic as a schematic and explain what
it shows. An ordinary point or
radius-graph preview must not silently stand in for the computed filtered
complex, and a coloured bar must not imply a verified source-cycle association.

End the example with a figure or compact table that answers its opening
question, the supported interpretation, and one next question. Extra views
belong in a short optional section once the main argument is complete.

## Reveal options when they become useful

Keep the cadence of a teaching notebook: one question, a small public call,
a visible result, and an interpretation. The first plotting cell should expose
the mathematical choices needed for that question. For example, after loading
CairoMakie, begin with:

```julia
OP.visualize(diagram; kind=:barcode, dim=1)
```

Explain `kind` as the view and `dim` as the homological degree. Use the notebook's
automatic display of its final expression; reserve `display(...)` for cells
that intentionally show several results. Separate data preparation from the
plotting call when nested constructors obscure what is being viewed.

Introduce an option at the first question it answers. When comparing two
barcodes, explain and pass the same `window` to both. When comparing empty,
partly occupied and full masks, keep `colorrange=(0, 1)` visible from the first
snapshot. A shared scale, orientation, degree, coefficient field or filtration
convention can determine the interpretation and must not be hidden as polish.
Short code is useful only when the mathematical assumptions remain clear.

Let established defaults carry routine presentation. Avoid repeating the
already selected backend, default image orientation, font sizes and canvas
sizes in each main-path cell. Use nearby prose and captions to explain the
figure. Keep a quantity label or title in the call when it conveys information
the default cannot infer, such as what the image values measure.

Put cosmetic customization and export in a short optional section after the
main result. Introduce `VisualStyle` once there and reuse it explicitly when
previewing and saving a figure. Later main-path cells must work if this entire
section is skipped. Avoid notebook-local plotting wrappers, near-duplicate
API aliases, global theme changes or keyword bundles introduced solely to
hide the plotting call; readers should be able to reuse the visible public API.

One canonical notebook supplies the website and the executed download. Mark
optional sections using the [publication workflow](README.md#optional-sections)
so the website can fold them without deleting their code, explanations or
saved results. Essential assumptions and required figures stay in the main
reading path. The downloadable notebook retains ordinary cells in reading
order, with an explicit optional heading.

Treat a useful default figure as a library requirement. Review the smallest
public call with no style or size override: readable labels, sensible margins,
restrained color, appropriate aspect and enough information to interpret the
result. Explain symbols actually used; do not print an entire diagnostic
catalog on a simple figure. Always retain visible notices for censoring,
omitted results, coordinate collisions and other relevant limitations.
Exact values and detailed display metadata remain inspectable. Repeated
cosmetic repairs in tutorials are a reason to improve the shared recipe.

Before publication, execute the full notebook and separately check that the
main path works with optional cells skipped. Review both the simple defaults
and the customized/exported figure, along with the collapsed and expanded
website section at desktop and narrow widths. Check the saved notebook as
well as the generated page; folding must never replace output capture.

## Review the reader's understanding

Before considering a page finished, check:

1. What question brings the reader here, and does the article's family fit it?
2. What object do we start with, and what do we learn or construct?
3. Is the relevant mathematical or practical context clear without forcing an unrelated learning sequence?
4. Are new terms, symbols, options, and returned results explained?
5. Does the example show why the answer makes sense?
6. Are the necessary assumptions clear and accurate?
7. For a lesson, does its next step follow from a question it raised? For a guide or reference, do contextual links support the task without imposing an artificial sequence?

Review related pages together for notation and continuity. Passing examples
and complete API listings are useful checks, but neither establishes that
the writing teaches its intended reader. Ask a reader with the stated
prerequisites to explain the result in their own words.
