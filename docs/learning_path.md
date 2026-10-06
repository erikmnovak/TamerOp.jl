# Implementing the first learning path

The [full documentation programme](documentation_plan.md) owns the overall
curriculum, complete mathematical-map relation and integration across the five
article families. This brief develops worked examples and teaching acceptance
for the first complete practical route through the documentation site:

**Ring example → why a second parameter changes the problem → finite
encoding → inspect spaces and maps → interpret a figure.**

Installation supports running the notebooks; it is not a prerequisite for reading
the lessons and their saved figures. The definitions offer an equally valid starting
point: **persistence modules → finite encodings**. Readers coming from the ring
may also take the optional branch from “Why two parameters?” through persistence
modules before continuing to finite encodings.

Readers can see the available routes on the [linked reading map](src/reading_map.md).
Show both starting points with equal visual prominence, label the optional
branch, and place installation beside the notebook as support. Reference guides
remain grouped by task instead of appearing as required stops. The diagram and
linked outline share `reading_map.toml`; repository treatments are labeled, and
unwritten lessons remain in the catalog and authoring plans rather than appearing
as working map links.

As this plan grows, apply the
[map design principles](writing.md#preserve-choices-as-the-reading-map-grows)
and the [expansion checklist](README.md#map-expansion-checklist). New lessons
must justify their place by the reader's question; the plan's completeness
must not turn the overview into a crowded catalog or a compulsory sequence.

Each lesson closes with one motivated primary continuation and at most one
alternative. `reading_map.toml` declares those destinations; the publication
build checks the closing prose and uses the same next destination in the footer.
After inspecting and interpreting the square, the computational route continues
to tameness. The theoretical route goes through indicator presentations first;
both reach practical tameness, then category-specific algebra. Supporting guides
remain references that readers can consult when their question arises.

Lessons contain no implementation status, development chronology, validation
reports, or general feature roadmaps. Clearly labeled figure placeholders may
describe a missing visual and its mathematical purpose. This brief and the testing/backlog
documents hold that material. Keep actual assumptions and operational constraints
in the lessons where readers need them.

The intended reader knows vectors, matrices, and the idea of a hole in a shape,
but need not know Julia, posets, or persistence modules. Follow the
[writing guide](writing.md). The existing [finite-encoding introduction](finite_encodings.md)
supplies the mathematical narrative; preserve it when adapting material into
the site. The briefs below describe teaching steps; square construction,
inspection and figure interpretation share one canonical notebook and site page.

The [ring notebook](tutorials/ring.ipynb) and the first local Documenter site
scaffold are now implemented. The [publication workflow](README.md) executes
the canonical notebook once, then generates a website lesson and an executed
download with ten static figures. Its 16 code cells include exact interval
and component assertions. Nine figures belong to the main lesson; one previews
the optional styling/export section. Short public plotting calls use the
package defaults; shared scales are introduced when comparing results.
The website folds the optional section while the notebook retains its cells.
Installation, both notebooks, the [two-parameter bridge](two_parameters.md),
and the existing mathematical chapters now have publication destinations and
navigation. Both static lessons and their saved notebook outputs have passed
publication and rendered review, as recorded below. Live WGLMakie controls
inside notebook frontends, first-path reference entries, reader walkthroughs
and deployment remain separate completion checks.

The bridge is a short entry point into the existing mathematical sequence,
not a replacement for those chapters. Readers can follow the practical route
from the ring through the bridge to the square, consulting the full
persistence-module definitions as needed. Readers wanting the mathematical
development can follow persistence modules → finite encodings → indicator
presentations → tameness → why finite computations stay tame. Both routes use
the same examples and the same canonical chapter sources. In particular, the bridge
reuses the chapter's parameter-order diagram; finite-encoding constructions
and their hypotheses remain in the existing chapters.

Five background chapters are available without requiring installation.
[Persistence modules over posets](persistence_modules.md) develops the ring,
spaces and maps, and comparable parameters for pages 2 and 3 below.
[Finite encodings](finite_encodings.md) follows with the square module,
recovery by pullback, and the relationship between the nine-label and
four-label representations for pages 4 and 5.
[Indicator presentations](indicator_presentations.md) explains the square's
input regions and coefficient matrix, then develops two overlapping square
summands with inclusion and projection maps. This supplies a richer example
for later exploration without replacing the first path's square.
[Tameness and scope](tameness.md) connects constant subdivisions, finite
encodings, and finite fringe presentations, and uses a module of constant
dimension one to show why controlling stalk dimensions alone is insufficient.
[Why finite computations stay tame](practical_tameness.md) then explains how
finite constructions supply the condition and how compatible algebraic
operations preserve it. Readers with the finite-encoding definition can also
enter that practical explanation directly before following the full theory.
Their static diagrams are available. The inspection notebook supplies static
parameter, poset and matrix views plus optional instructions for linked live
selection. Following a chosen vector through a multiparameter module and
linking general multiparameter classes to source representatives remain planned.
Ordinary persistence already retains optional cycles, finite-death filling
chains and scale-specific cocycles; that is a separate, supported workflow.

The square's inspection lesson now has a
[canonical teaching notebook](tutorials/inspect_encoding.ipynb). It follows
original parameters into the actual finite poset, checks stalks and maps, and
exports the selected endpoint-map figure through the package API. Two overlapping squares
show why nonzero successive maps can have zero composite. A live WGLMakie
session lets the reader change the selection and switch between module
and presentation views. The revised lesson captures the required figures with
CairoMakie. Optional Markdown examples explain live inspection and a session
snapshot; a separate optional section exports PNG and SVG directly from the
encoding, without creating a session. Publication executes all code cells,
including these static exports; it preserves the live examples without running them.
The revised publication was checked on 2026-10-04: all 30 code cells executed,
all 11 static figures were captured in the site and notebook download, and the
main path passed with optional sections omitted. The existing square and
presentation oracles passed 8,710 and 43,134 assertions, respectively. Browser
review covered desktop and narrow layouts, light and dark themes, optional
sections, equations, and saved outputs in JupyterLab. The final publication
used an isolated working-source copy while other library work continued;
its package and environment fingerprints are retained in the build record.

These are static-publication checks. Earlier native and standalone-browser
inspector checks retain their own scope; interactive WGLMakie controls inside
a notebook frontend still require separate acceptance. The source supplies
teaching steps 4–6 below, and `publication.toml` connects it to the ring-to-square
route.

## Keep the mathematical chapters distinct

The short bridge and the full foundations are alternative entry routes. The
later chapters answer different questions about the same finite description;
they should link to established explanations instead of repeating their proofs.
The [reading map](src/reading_map.md) shows suggested continuations, while this
table records the boundaries authors should preserve.

| Treatment | Reader's question | Owns this explanation | Leaves to the next treatment |
| --- | --- | --- | --- |
| [Two parameters](two_parameters.md) | Which parameters can we compare? | A short route from the ring to partial order, spaces, and maps | Full module axioms and finite-encoding constructions |
| [Persistence modules](persistence_modules.md) | What are spaces and maps? | The full definition, composition, and why dimensions lose information | How finite data recover an infinite parameter family |
| [Finite encodings](finite_encodings.md) | How can finite data recover a module? | The square's classifier, finite poset, spaces, maps, and recovery | Constructing presentations and proving general existence |
| [Indicator presentations](indicator_presentations.md) | How do regions and a matrix define spaces? | Upsets, downsets, active matrix blocks, images, and induced maps | The equivalence between finite descriptions |
| [Tameness and scope](tameness.md) | When do finite descriptions exist? | Constant subdivisions, finite encodings, finite fringe presentations, and the counterexample | Why standard finite constructions satisfy the condition and preserve it |
| [Why finite computations stay tame](practical_tameness.md) | Why do finite computations stay finite? | Finiteness supplied by construction, closure under compatible maps, and the relevant abelian-category result | Category-specific interpretations of derived computations |
| [Categories and derived computations](math_categories.md) | In which category does this calculation take place? | The package's actual finite-base operations and comparison hypotheses | Operation-specific guides and reference entries |

### Follow-on brief: why finite computations stay tame

- **Source:** [practical_tameness.md](practical_tameness.md). Its normal place is
  after tameness; an optional entry from finite encodings answers the practical
  question early without making category theory a prerequisite for the first
  computation.
- **Prerequisites:** Finite encodings and basic matrix kernels and images.
  Recall the needed meaning of tameness and explain an abelian category through
  the operations it permits.
- **Starting objects:** A filtration by subcomplexes of a fixed finite complex,
  and the square module already used throughout the foundations.
- **Content:** Show how finite input supplies an encoding of a complex and its
  differentials. Follow kernels, images, quotients, and homology through that
  encoding. Use the square map `M ⊕ M → M` with matrix `[1 1]` as a small
  calculation, and introduce the precise abelian-category theorem only after
  the reader understands the closure question. State the allowed regions and
  morphisms alongside any theorem that depends on them.
- **Expected conclusion:** An established finite construction can discharge
  the tameness hypothesis. Finite encodings of two objects alone do not justify
  arbitrary morphism or closure claims, and the conclusion concerns the
  represented model on its stated domain.
- **Visual checkpoint:** A static diagram should connect the finite complex,
  its boundary maps, and the encoded homology; the square calculation should
  expose the actual kernel and image. Captions distinguish mathematical
  schematics from package output.
- **Transition:** Once these operations remain finite, which category does
  TamerOp use for Hom, Ext, and Tor? Link the category guide without suggesting
  that exact recovery or abelian closure establishes ambient derived
  equivalence.
- **Acceptance:** Check the square calculation by hand, verify the theorem's
  object and morphism hypotheses against its primary source, and retain the
  distinction between a finite computational model and an arbitrary continuum
  module. A short program or finitely many sampled queries is not itself a
  finite-encoding proof.

## Page briefs

### 1. Install and reopen a working environment

- **Destination:** `start/install.md`.
- **Reader's question:** How do I install TamerOp and return to the same work later?
- **Prerequisites:** Ability to open a terminal or the Julia application.
- **Starting object:** An empty working directory and a supported Julia installation.
- **Content:** Adapt the README's installation, project activation, import, and
  reopening instructions. Distinguish terminal commands from Julia commands;
  explain the purpose of `Project.toml` and `Manifest.toml`. CairoMakie supports
  both lessons' static figures; WGLMakie belongs only to the square's optional
  live instructions. Verify the distribution instructions when publishing.
- **Expected conclusion:** The reader can load `TamerOp` as `OP` in their own
  environment and reopen it without reinstalling the package.
- **Transition:** What small calculation has an answer I can recognize?
- **Acceptance:** A fresh environment loads the published package through these
  instructions. Local package loading alone does not satisfy this check.

### 2. Follow a hole through the ring example

- **Destination:** `tutorials/ring.md`.
- **Reader's question:** What does a persistence interval tell me about a shape?
- **Prerequisites:** Page 1; no homology terminology assumed.
- **Starting object:** The matrix `[0 0 0; 0 5 0; 0 0 0]` interpreted as appearance
  grades of square top cells in a nonperiodic cubical sublevel filtration.
- **Content:** Explain the outer ring at grade 0 and the filled center at grade 5,
  then introduce a homology class, birth, death, and degree. Use the direct
  `cubical_persistence` route over F₂ and inspect intervals and provenance.
- **Expected conclusion:** There is one H₁ interval `[0,5)` and one essential
  H₀ class born at 0. The right endpoint is excluded because the hole is already
  filled at 5. This call returns a persistence diagram, not an `EncodingResult`.
- **Transition:** What changes if a second, independently varying parameter
  influences which cells are present?
- **Acceptance:** Predict and verify that changing the center's grade to 3 gives
  `[0,3)` while the essential component still begins at 0.

### 3. Explain why a second parameter changes the problem

- **Destination:** `explanations/two_parameters.md`.
- **Source and status:** [two_parameters.md](two_parameters.md), implemented
  as a short static explanation on 4 October 2026. The publication manifest
  places it after the ring; it joins the existing finite-encodings chapter.
- **Reader's question:** Why can I no longer arrange all parameter choices on one line?
- **Prerequisites:** Page 2; comparisons of pairs of real numbers.
- **Starting object:** A schematic two-parameter filtration with its coordinate
  directions explicitly chosen so that increasing either coordinate gives an
  inclusion. A density cutoff with the opposite convention needs reorientation.
- **Content:** Introduce coordinatewise order through two comparable pairs and
  an incomparable pair. Explain vector spaces and structure maps before naming
  a persistence module. Show why counts of classes do not specify the maps.
  Avoid implying that the ring's direct barcode computation has secretly
  constructed a bifiltration or that every multiparameter module has a barcode.
- **Expected conclusion:** The reader can decide when a parameter pair carries
  a structure map and explain why retaining dimensions alone loses information.
- **Transition:** Can infinitely many spaces and maps have a finite description?
- **Acceptance:** The reader identifies `(1/4,3/2)` and `(3/2,1/4)` as incomparable,
  although both lie in the square used on the next page.
  The local ten-page publication build passes link, anchor, image, and notebook
  checks. Chromium review confirms rendered equations and the shared diagram,
  no overflow at 1280px and 390px, and working ring → bridge → finite-encodings
  links. Reader explanation of the exercise remains a separate acceptance check.

### 4. Construct a finite encoding of the square module

- **Destination:** Construction section of `tutorials/inspect_encoding.md`,
  generated from the canonical inspection notebook.
- **Reader's question:** What finite object can recover this module on ℝ²?
- **Prerequisites:** Page 3; the meaning of an identity and a zero linear map.
- **Starting object:** Exactly the closed square-supported module from
  `finite_encodings.md`, with coefficient field ℚ and support `[0,2]²`.
- **Content:** First work through the spaces and maps below. Revisit the valid
  nine-region Cartesian model from the introduction. Then express the module
  as the image of the scalar `[1]` between one birth upset and one death downset,
  explaining those terms through their regions before using the constructors.
  Construct an actual `EncodingResult` and identify its finite poset, finite
  module, and assignment from original parameters. Explain the returned
  representation; do not demand nine labels or particular numeric label IDs.
- **Expected conclusion:** The reader can explain `M ≅ M_P ∘ π` as recovery of
  both spaces and maps. Two different finite encodings can recover this same
  module. Connect this finite-description viewpoint to Ezra Miller's theory
  without claiming every abstract tame module has an implemented encoder.
- **Transition:** How can I check what the result says at particular parameters?
- **Acceptance:** Verify the closed boundaries, the interior identity map, and
  a map leaving the support. Display the actual returned labels and their order.

### 5. Inspect spaces and maps in the returned representation

- **Destination:** `tutorials/inspect_encoding.md`.
- **Reader's question:** How do I recover a space or a map at original parameters?
- **Prerequisites:** Page 4; matrix sizes and multiplication.
- **Starting object:** The same square `EncodingResult`; do not reconstruct a
  disconnected example or substitute an identity encoding of a finite input.
- **Content:** Display the returned encoding, check its coefficient field through
  `provenance`, and use `encoding_poset`, `encoding_map`, `dimensions`, and
  `encoding_module`. Explain that `dimensions(enc)` gives the stalk-dimension
  vector indexed by finite labels. Distinguish the root workflow from advanced
  `locate` and `structure_map` queries. Explain source columns and target rows.
  Inspect maps as well as dimensions; in the two-square variation, compare the
  composite of the adjacent maps with the direct map. The optional presentation
  section follows `encoding_presentation` into the active
  downset rows and upset columns. It explains why the image of the restricted
  matrix is the stalk, explicitly requests embedded image bases, and checks
  the induced-map equation and the two-square zero composite. An active-zero
  block distinguishes support membership from a nonzero coefficient.
- **Expected conclusion:** The reader can recover an interior identity and a
  correctly shaped zero map, and can distinguish order in ℝ² from order among
  finite labels. Equal labels do not imply that the original points are comparable.
- **Transition:** How can a figure make this finite model easier to understand?
- **Acceptance:** Explain the table below before running its queries. An exercise
  uses an incomparable pair and asks why there is no structure map to query in
  the original module, even if both points receive the same finite label.

### 6. Interpret a figure of the same encoding

- **Destination:** Figure sections of `tutorials/inspect_encoding.md`, generated
  from the same notebook as construction and inspection.
- **Reader's question:** Which parts of the module does this picture show, and
  what do I still need to query?
- **Prerequisites:** Step 5; CairoMakie from the installation guide. Introduce
  WGLMakie only for optional live inspection.
- **Starting object:** The same verified square encoding and its query results.
- **Content:** Prepare a parameter-plane support picture and a finite-poset
  diagram using labels obtained from the result. Mark the square's included
  boundary and show the spaces and representative maps. Reuse labels across
  panels. Explain that a dimension heatmap alone does not display the maps.
  Introduce `visualize` with each mathematical selection; use `visual_spec` and
  `visual_metadata` to distinguish an incomparable pair from a zero map.
  After checking the mathematics, offer live `inspection_session` and
  `inspection_snapshot` instructions in Markdown. Explain exact parameter
  selection, preservation of the query between module and presentation views,
  approximate pointer coordinates and the running Julia requirement. Keep basis
  computation an explicit single-stalk choice. A separate optional export section
  uses `save_visual` directly on the encoding with the endpoint pair selected,
  so it works without the live section.
- **Expected conclusion:** The reader can trace a parameter through its label
  to a space, interpret an arrow, and state what the figure omits. They can
  save a figure and reopen the canonical notebook.
- **Transition:** How would data give rise to an encoding, or how would a module
  with larger spaces and nontrivial matrices behave? These are later paths.
- **Acceptance:** Executed notebook, downloaded notebook, and published figure
  agree. Captions state the field, domain, closed support, and meaning of arrows.
  Static A42a/A83 recipes and exports have passed native rendering review.
  An earlier notebook version's 23 code cells passed headless display capture;
  its selected-state exports also passed static visual review. All 550
  inspector assertions pass across five fields, including native callback and
  lifecycle checks; 116 additional server-embedding checks pass. The author
  reports successful manual acceptance of the local live-server two-square
  inspector, including layout, exact queries, pointer/keyboard controls, view
  changes, error recovery, reset, reload, linked tabs and closure. Actual
  notebook-frontend acceptance remains separate; captured notebook displays
  do not establish that integration.

## Visual checkpoints for notebook publication

The first path keeps its existing sequence and mathematical examples. Apply
the [notebook writing guidance](writing.md#plan-the-visible-result-of-a-notebook)
by planning the visible result alongside each question. Both canonical notebooks
now belong to the publication build. These rows specify the figures and
interpretations to review in their generated pages and executed downloads.

| Lesson | Required visible result | Prediction or interpretation to check |
| --- | --- | --- |
| Ring | Input top-cell grades; active-cell masks at grades `-1`, `0`, and `5` with the same orientation; the degree-one barcode and diagram, with the essential degree-zero interval explained separately | The complex is empty at `-1`, has one hole at `0`, and has filled that hole at `5`; the degree-one interval is `[0,5)`. These masks describe the stated cubical fixture, not an arbitrary complex renderer. |
| Change one input | Original center grade `5` and changed grade `3`, with directly comparable barcodes | The hole dies earlier, giving `[0,3)`; the essential component still begins at `0`. Ask for the prediction before showing the changed result. |
| Second parameter and square | A small parameter-plane diagram with comparable and incomparable pairs; the actual returned finite-poset view alongside a selected stalk/map | A two-dimensional dimension plot does not determine the maps; nine illustrative regions are not a required encoder output. |
| Overlapping squares | Static selected-map views for `a → b`, `b → c`, and `a → c`, retaining the same parameter window and labels, with the three small matrices displayed explicitly | Both adjacent maps have rank one, but their composite is zero. Check all three answers together in the generated page and download. |
| Indicator presentation | The overlap's active block and image basis as displayed matrices; a figure of the active-zero stalk; a figure of the `b → c` presentation map with its bases and ambient projection | Presence in the supports is insufficient: the active block can be `[0]`. An empty image basis has a mathematical meaning, and the induced map satisfies `B_c C = R B_b`. |
| Live inspection and export | Optional Markdown instructions for a live selection and snapshot; separate PNG/SVG exports of the selected endpoint map directly from the encoding | Identify what the static figure retains and which interaction needs live Julia. Publication executes the direct exports without creating a live session; the session snapshot example and live notebook acceptance remain separate. |

Use image, interval, module and presentation recipes already available for the
first pass. A full filtered-complex viewer, automatic cross-panel annotations,
or a verified source-cycle overlay is not a prerequisite for publishing the
ring and square lessons. If a planned view needs one of those capabilities,
describe the coming figure and its purpose explicitly; do not substitute an
unlabelled surrogate.

The publication manifest includes both canonical notebooks. Their source files
may have cleared outputs; the build must execute the code and capture all
declared static figures before producing pages and downloads. Earlier headless
execution and separately reviewed exports remain evidence only for the versions
and contexts tested. Check captions, mathematical answers and images together
in a fresh build of the revised square lesson, including the optional static
export and the main path with optional cells skipped. Live instructions remain
Markdown examples, so static publication does not establish live notebook
acceptance. Per-build execution evidence for both lessons belongs in the
generated `downloads/publication.json` described in the build guide.

## Mathematical specification before tutorial cells

Fix coordinatewise order on ℝ² and coefficient field ℚ. Let
`U = {q : q₁ ≥ 0 and q₂ ≥ 0}` and `D = {q : q₁ ≤ 2 and q₂ ≤ 2}`.
Take the image of the scalar-one map from the indicator module of `U` to that
of `D`. Its stalk at `q` is ℚ precisely on `U ∩ D = [0,2]²`, and is zero
elsewhere. For `q ≤ r`, its map is `[1]` if both points are inside the square;
otherwise it is the unique zero matrix of size `dim(M(r)) × dim(M(q))`.

This rule respects composition: if both ends of a comparable triple lie in
the square, the middle point does too. In every other case the direct map
and the composite are zero. Endomorphisms of a zero space are `0 × 0` matrices
and serve as its identity. These statements specify the mathematical object
before any label numbering or executable lesson is chosen.

| Query | Expected answer | Reason |
| --- | --- | --- |
| Stalk at `(-1,1)` or `(3,1)` | dimension 0 | Outside the support |
| Stalk at `(0,0)`, `(2,2)`, `(0,2)`, or `(2,0)` | dimension 1 | Both boundaries are included |
| `(1/4,1/2) → (1,3/2)` | `1 × 1` identity | Comparable interior points |
| `(-1,1) → (0,1)` | `1 × 0` zero matrix | Entering the support |
| `(1,3/2) → (3,3/2)` | `0 × 1` zero matrix | Leaving the support |
| `(-1,1) → (1,1) → (3,1)` | Composite equals the direct `0 × 0` map | Functoriality through a nonzero space |
| `(1/4,3/2)` and `(3/2,1/4)` | No comparison in either direction | A shared label cannot add an ambient comparison |

For the oracle's `backend=:pl_backend, poset_kind=:signature` configuration,
explain labels by the
two bits `u(q) = [q ∈ U]` and `c(q) = [q ∉ D]`. Both bits are order preserving.
The realized signatures are `(0,0)`, `(1,0)`, `(0,1)`, and `(1,1)`, ordered
coordinatewise. Only `(1,0)` has a one-dimensional space. This gives a diamond
with four labels; it is a different valid encoding from the illustrative
nine-label grid. Numeric IDs belong to the returned representation and should
be discovered with `locate`, never used as the definition of a region.
Exterior points still receive valid labels with zero-dimensional stalks:
the square is the support, not a restriction of the query domain. A label can
also represent a disconnected set of parameters, rather than one rectangle.

The checker verifies that characterization for the selected encoder. A future
encoder could use a different finite poset: update the explanation of its
representation while retaining the stalk, map, boundary, and composition
requirements. Finite sampling checks the implementation on the example; the
argument above explains the module over its entire domain.

## After the first encoding: the mathematical curriculum

The first route establishes an object we can continue to study: a finite
poset, its spaces and maps, and the classifier relating them to the parameter
domain. Mathematical lessons should develop what happens **before** that
description and what becomes possible **after** it. The post-encoding material
is not reserved for library guides. Its purpose is to explain new mathematical
questions, objects and conclusions, with the same approachable prose, short
computations and interpreted figures as the introductory lessons.

The [article inventory](article_inventory.toml) owns the titles, scopes and
editorial actions. The stable IDs below identify planned treatments; they are
not links to finished articles. This brief describes their intended teaching
relationships rather than a second article-status list.

| Branch | Proposed teaching relationships | What the reader should understand |
| --- | --- | --- |
| Modules and their maps | Start with `diamond_maps`; branch into `module_operations`, `hom_spaces` and `universal_constructions`. | A map between modules must respect structure maps; pointwise linear algebra must assemble into a compatible object. Combining modules and imposing agreement are different operations. |
| Resolutions and derived algebra | Use maps and exact sequences to motivate `resolutions_lesson`; develop `module_complexes` as needed before `derived_functors`, then optional `products_lesson` and `complexes_spectral_sequences`. | Resolutions and complexes make new questions computable. Derived groups, products and page diagrams require their category, grading and interpretation, not just a returned dimension. |
| Changing the description or base | Begin `change_of_posets` after understanding the classifier. Add `pushforwards_lesson` using the diagrams from `universal_constructions`; consult `math_categories` for comparison hypotheses. | Restriction, refinement and pushforward do different things. Comparison maps need not be isomorphisms; retaining the ambient module does not identify every finite-category derived result with an ambient one. |
| Invariants and summaries | Begin `slices_invariants` directly after the square, then choose `slice_barcodes`, `signed_summaries`, `generalized_rank_lesson` or `support_geometry_lesson` according to the question. | Dimensions, ranks, restrictions, signed reconstructions and region measurements retain different information. The classifier matters for geometric measurements. Advanced algebra is not a prerequisite for this branch. |
| Support and decompositions | After module maps and sums, develop `algebraic_support` and `decomposition_lesson`; consult resolutions for the Betti/Bass part. | Nonzero support, generators, terminal classes, genuine direct summands, signed contributions and approximate tracks are distinct descriptions. |
| Comparisons and numerical features | From a chosen invariant or slice, enter `distances_stability` or `features_lesson`; these are related choices rather than a compulsory order. | Identify what is compared and what sampling, smoothing, normalization and vectorization preserve or discard. State the hypotheses for any stability claim. |
| Interpreting pictures | Enter `reading_module_figures` after the square for its basic views; revisit its optional synthesis after selected invariant or feature lessons. | A figure represents a particular object or transformation. Poset layout, classifier geometry, dimension colors, intervals, signed weights and feature pixels support different conclusions. |

These are suggested branches, not a demand to read every article in a row or
every predecessor before continuing. Each authored lesson should state the
knowledge actually needed for its example. In particular, the invariants and
features route remains available without the derived-algebra branch.

Use the square and the two-square variation wherever they show the phenomenon
clearly. A small diamond is useful for branching maps; a short exact sequence
or module pair may be needed to make an algebraic distinction visible. Introduce
that object and the expected behavior explicitly. Do not force an uninformative
example to carry every new construction.

Visualization accompanies all branches. The interpretation lesson gives a
later synthesis, with optional deeper sections; it does not postpone figures
until after vectorization or make every reader study every kind of plot.
Its boundary with the library guide is mathematical meaning versus selecting
views, controlling interaction and exporting files.

Keep the article inventory's `related` entries as associations. When a lesson
is authored and reviewed, add its actual continuation choices to
`reading_map.toml` and align its closing prose. Do not add links to unwritten
pages. Extend the main diagram with a few meaningful branch choices and use
focused branch views if detail becomes crowded. The separate topic map can
gather lessons, usage guides, reference, implementation and reports about each
area without turning them all into stops in this learning progression.

The next mathematical continuation to draft is `diamond_maps`: it makes the
transition from inspecting maps inside one module to constructing maps between
modules. The independently approachable `slices_invariants` branch can follow
without waiting for the more advanced algebra lessons.

## Executable evidence and authoring order

The self-contained [mathematical oracle](build_scripts/check_first_encoding.jl)
checks the ring and square through public package APIs. It is a validation
fixture, not a second independently authored tutorial. It includes boundary
and exterior points, all comparable pairs and triples in its sample, and an
incomparable pair with a shared label. Run from the repository root:

```sh
julia --startup-file=no --project=. docs/build_scripts/check_first_encoding.jl
```

The verified Julia 1.12.1 run passed 8,710 checks, including 49 square query
points, 784 comparable-pair maps, and 7,056 compositions. These are checks of
the stated example, not a claim about all encoders or arbitrary modules.

The [indicator-presentation oracle](build_scripts/check_indicator_presentations.jl)
also checks the square and its two-summand extension. Its Julia 1.12.1 run
passed 43,134 checks: 49 one-square points and 81 two-square points, all
comparable-pair maps and compositions in those samples, and agreement with
the prescribed summands in compatible bases. It verifies the inclusion,
projection, and zero composite described in the third background chapter.
Run it with:

```sh
julia --startup-file=no --project=. docs/build_scripts/check_indicator_presentations.jl
```

Keep installation verification, mathematical checks, rendering checks, and
reader review separate. The background chapters, mathematical oracles, and
square inspection notebook are available. The site publication manifest and
navigation include both notebooks and the linked ring-to-square route.
The revised square publication has passed execution and rendered review,
including its saved outputs in JupyterLab. First-path reference prose,
live-widget notebook acceptance, reader review, and deployment remain separate
work.

1. Reconcile the [API inventory and backlog](api_coverage.toml); use the selected
   first-path bindings to constrain the first reference-writing pass.
2. Extend the documentation environment and site scaffold already added for
   the ring. Stage further explanations with links and stable anchors where
   needed, preserving their mathematical qualifications.
3. Preserve the completed square publication as the route grows. Generate
   displayed cells, outputs, and downloads from its canonical source; recheck
   selected maps, optional sections and the continuation into tameness when
   those parts change, without maintaining competing examples.
4. Write only the reference entries needed for this route initially. Include
   the specific method families, field/grade conventions, options, returned
   objects, and meaningful errors. Keep other families in the manifest backlog.
5. Execute from a clean documentation environment, check installation separately,
   render and inspect the figure, then ask a reader with the stated prerequisites
   to explain the result. Publish the first path when these checks pass; the
   remaining reference backlog does not block it.

### Generalized rank and GRIL: planned teaching branch

After the reader can follow maps on the finite encoding, `generalized_rank_lesson`
should ask what persists through an entire branching region. Use the two-source
fork with coincident versus transverse images: all stalk dimensions and pairwise
ranks agree, but the generalized ranks differ. Introduce compatible vectors and
the quotient identifying transported vectors through that calculation. Place
connectedness, convexity and the finite-label interpretation beside the first
query, before moving to ambient regions.

Carry the branch into a signed finite-family example with coefficient -1, then
an expanding continuous worm. Pair the worm with a rank-versus-width staircase
and mark its supremum; use exact rectangle and quadrant answers to explain
levels, lengths, endpoint attainment and the grid extension's boundary behavior.
The viewer should distinguish a signed reconstruction from a decomposition,
and an exact width at selected probes from a complete description of the module.
Keep these figures mathematical; no new visualization API is a prerequisite.

The separate `guide_generalized_rank` explores query choices, witnesses,
validation, resource bounds and feature-vector use. B54 explains the actual
constraint/quotient solver, fiber contraction and critical-width algorithm.
Its authored source is `docs/implementation/generalized_rank.md`; runnable
examples live in `docs/examples/generalized_rank.jl`. The catalog owns source
presence and article identity. Add lesson routes and public navigation only
when those articles are authored and published. Learned selection and
end-to-end differentiation remain the separate P3 research addition A122;
fixed user-chosen GRIL probes are within A54.


### Ordinary results as an optional analysis branch (A116)

After the ring, readers who want a comparison can take `ordinary_analysis`,
the canonical notebook at `docs/tutorials/ordinary_analysis.ipynb`. Its three
points give two mergers that can be predicted before computation. Carry that
same result through a moved point, bottleneck/Wasserstein comparison, an explicit
essential-bar choice, tents, a sampled image and a saved result with a retained
cocycle. This is an optional practical branch; it does not add a prerequisite to
the main finite-encoding route.

The existing `ordinary_persistence` library guide owns the operational choices.
Keep the planned mathematics in `distances_stability` and `features_lesson`:
diagonal/essential matching, reflection of superlevel time, multiplicities,
tents versus weighted averages, Gaussian grid sampling and information lost by
features. B39 explains matching and numerical assignment, B43 explains feature
computation, and B48 explains the owned exact-data schema and its validation
limits. Expand those existing implementation identities instead of creating a
second ordinary-analysis algorithm chapter. Reference contracts remain assigned
by `api_coverage.toml`. The notebook source is available now; website routing and
rendered publication must be added and verified separately before advertising a
public lesson route. The article catalog records that distinction.

## Teaching the next computational families

The [full documentation programme](documentation_plan.md) and its complete
lesson relation now govern this expansion, including visual mathematics.
The [catalog](article_inventory.toml) owns individual scopes. These example
briefs supplement that plan; they are not a second inventory or execution order.
Every published mathematical article belongs in a focused map view and the
shared outline. Keep the introductory diagram small and branch ordinary
variants directly from filtration mathematics where appropriate.

- **Inputs and variants.** In `image_lesson`, use a hollow voxel block and its
  filling to distinguish top-cell grades from vertex samples; the `voxel_cavity`
  recipe predicts its interval before executing. In `relative_extended_lesson`,
  an interval/endpoints or disk/boundary example introduces quotient chains and
  a connecting map. Then `extended_persistence_lesson` follows an ordinary-to-
  relative filtration with typed intervals: replacing infinity by a cap does
  not give the same construction. The ordinary guide explains the supported
  practical choices without absorbing either mathematical lesson.
- **Algebra from actual maps.** In `diamond_maps` and `module_operations`,
  propagate a supplied vector on a branching module, construct its generated
  submodule and inclusion, and test whether a submodule splits. The guide and
  recipe use the same witness. A two-square sum supplies known multiplicities;
  a bipath example retains its gluing maps. Universal constructions have their
  separate pullback/pushout question.
- **Resolutions and induced information.** Compare a short ordinary resolution
  with a bounded hook-family rank-exact calculation. Explain maps, category and
  exact structure before reading its table. For morphism-induced matchings,
  follow identity and zero maps with a pair for which image data alone do not
  determine the additive matching. A bottleneck optimizer answers another
  question. Cheap selected outputs remain visible in the companion guides.
- **Cohomology and time.** Pair a cocycle with a cycle before using cohomological
  products or coordinates. Equal graded dimensions with different multiplication
  motivate persistent products. Circular coordinates require their own lift,
  solve and gauge explanation. A tiny add/delete sequence teaches temporal
  arrows and hand-derived intervals; its all-forward case checks agreement
  with ordinary persistence. Incremental reuse and certified correspondence
  receive separate explanations.
- **Certified approximation and richer summaries.** Check both interleaving
  equations on an exact and a failed supplied witness. Distinguish geometric
  sparsification, interval-support approximation and a pruned subquotient.
  A contour example should expose nonadditivity of stable rank. HN slopes and
  retained weights need a central-charge example; Jordan blocks need a separate
  ranks-of-powers example with coefficient assumptions stated locally.
- **Features, uncertainty and selection.** A few signed atoms give a checkable
  transport and explicit-grid convolution. Aligned replicates distinguish
  descriptive spread from confidence claims. Compare fixed GRIL probes with
  eventual learned selection; differentiation, ties and training claims require
  the supported P3 contract. B71 owns that computation and B72 owns ensemble
  alignment/statistical calculations, with individual transforms in B43.

Use diagrams to expose the mathematical distinction: actual voxel cells,
commuting maps, quotient/connecting maps, typed intervals, a short zigzag,
a cocycle pairing or a failed witness. Static figures can teach these ideas
before the corresponding interactive viewers exist. Caption what is computed
and preserved; keep common session/export mechanics in the shared usage guides.

Keep exploratory computations notebook-first. Small examples need independent
expected answers, maps and hypotheses, not only plausible figures. The optional
Python recipe starts with one array and checks equality with native Julia.
Publish coherent groups when their source, execution, editorial and site checks
are ready. Internal performance changes extend their algorithm accounts and
measured evidence without creating a new lesson per optimization.
