# Writing documentation that teaches

TamerOp's documentation should help readers understand why finite encoding
is the central object and how to use it to answer mathematical questions.
A reader should be able to follow the argument, interpret a result, and
choose a sensible next step.

Two principles govern documentation contributions: develop a connected
finite-encoding narrative, and teach in approachable language. Apply them
to tutorials, mathematical explanations, reference entries, and figures.

The [first learning-path brief](learning_path.md) records the initial page
sequence, reader questions, and expected mathematical answers. The
[API inventory guide](api_inventory.md) explains how to maintain the generated
binding inventory and the explicit reference-writing backlog.

The [reader-facing map](src/reading_map.md) makes these routes visible. When
adding a teaching page, record its question and suggested continuations in
`reading_map.toml`, using its canonical source path. Keep the graph acyclic,
and distinguish suggested routes from mandatory prerequisites. The site uses
one graph for the linked diagram and text outline, and labels repository
destinations separately. Add unwritten lessons to the backlog; add their
cards when the treatments exist. See the [map maintenance guide](README.md#maintain-the-reading-map)
for layout and browser checks.

End a lesson with a motivated next question and one primary continuation.
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
misleading impression. Preserve these principles when adding or moving pages:

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

Each substantial page should answer a question that the earlier explanation
has made meaningful. Say what object we begin with, what we learn or
construct, and why the next question follows. Carry a worked example across
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

1. Why is the reader encountering this topic now?
2. What object do we start with, and what do we learn or construct?
3. How does this fit into the finite-encoding workflow?
4. Are new terms, symbols, options, and returned results explained?
5. Does the example show why the answer makes sense?
6. Are the necessary assumptions clear and accurate?
7. Does the next step follow from a question the page has raised?

Review related pages together for notation and continuity. Passing examples
and complete API listings are useful checks, but neither establishes that
the writing teaches its intended reader. Ask a reader with the stated
prerequisites to explain the result in their own words.
