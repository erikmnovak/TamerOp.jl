```@raw html
<div class="home-page">
<div class="home-opening">
<header class="home-intro">
<p class="home-eyebrow">Multiparameter persistence in Julia</p>
```

# Keep the module, make its description finite

When a shape changes with scale, density, or several choices at once, which
features persist? How do they relate? TamerOp gives these questions a common
mathematical home.

The central idea is a **finite encoding**: a finite model of a persistence
module's vector spaces and linear maps, together with the link back to the
original parameters. Keep that structure available, and more questions become
possible.

```@raw html
</header>
<nav class="home-starts" aria-label="Ways to begin">
  <p class="home-eyebrow">Choose your starting point</p>
  <a class="home-start" href="tutorials/ring.html">
    <span class="home-start-kind">Start with an example</span>
    <strong>A hole appears and disappears <span aria-hidden="true">↗</span></strong>
    <span>Watch a shape change, then read its barcode.</span>
  </a>
  <a class="home-start" href="persistence_modules.html">
    <span class="home-start-kind">Start with definitions</span>
    <strong>Persistence modules <span aria-hidden="true">↗</span></strong>
    <span>Build the idea from spaces and the maps between them.</span>
  </a>
  <p class="home-setup">Read the lessons and saved figures without Julia.
    To run them, <a href="start/install.html">install and open a notebook</a>.</p>
</nav>
</div>
<p class="home-returning">Already have an object?
  <a href="guides/spaces_and_maps.html">Explore its spaces and maps <span aria-hidden="true">→</span></a>
</p>

<figure class="home-workflow" aria-labelledby="home-workflow-title home-workflow-caption">
  <div class="home-figure-heading">
    <p id="home-workflow-title">The finite-encoding workflow</p>
    <span>Many inputs. A shared mathematical object. Many questions.</span>
  </div>
  <div class="home-flow">
    <div class="home-flow-inputs">
      <p class="home-flow-label">Begin with</p>
      <div class="home-flow-item"><strong>Data</strong><span>Points, graphs, images</span></div>
      <div class="home-flow-item"><strong>Filtered complexes</strong><span>Cells and their grades</span></div>
      <div class="home-flow-item"><strong>Presentations</strong><span>Regions and matrices</span></div>
    </div>
    <div class="home-flow-arrow" aria-hidden="true"><span>prepare<br> and encode</span><b>→</b></div>
    <a class="home-model" href="finite_encodings.html" aria-label="Understand finite encodings through the square example">
      <strong>A finite encoding</strong>
      <span class="home-model-subtitle">Spaces, maps, and parameter lookup</span>
      <svg viewBox="0 0 350 190" role="img" aria-labelledby="home-square-title home-square-desc">
        <title id="home-square-title">From the square to a finite model</title>
        <desc id="home-square-desc">A point inside a closed square is assigned to the highlighted vertex of a four-vertex diamond. This vertex carries the one-dimensional rational vector space; the other three carry zero spaces. The diamond arrows point upward. The dotted line follows the parameter into its finite label.</desc>
        <defs>
          <marker id="home-order-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" class="home-arrowhead"/></marker>
        </defs>
        <g class="home-grid" fill="none">
          <path d="M42 25V144 M78 25V144 M114 25V144 M18 56H145 M18 92H145 M18 128H145"/>
        </g>
        <path class="home-axis" d="M18 25V144H145" fill="none"/>
        <rect class="home-square" x="42" y="56" width="72" height="72"/>
        <path class="home-lookup" d="M78 92 C140 92 149 92 206 92" fill="none"/>
        <circle class="home-point" cx="78" cy="92" r="5"/>
        <text class="home-point-label" x="67" y="78">q</text>
        <g class="home-order" fill="none" marker-end="url(#home-order-arrow)">
          <path d="M260 139L225 104"/><path d="M276 139L310 104"/>
          <path d="M224 80L258 44"/><path d="M310 80L277 44"/>
        </g>
        <circle class="home-zero-node" cx="268" cy="32" r="16"/>
        <circle class="home-active-node" cx="216" cy="92" r="17"/>
        <circle class="home-zero-node" cx="320" cy="92" r="16"/>
        <circle class="home-zero-node" cx="268" cy="151" r="16"/>
        <g class="home-space-label" text-anchor="middle" dominant-baseline="central">
          <text x="268" y="32">0</text><text x="216" y="92">ℚ</text>
          <text x="320" y="92">0</text><text x="268" y="151">0</text>
        </g>
        <g class="home-panel-label" text-anchor="middle"><text x="81" y="183">original parameters</text><text x="268" y="183">finite model</text></g>
      </svg>
      <span class="home-model-note">The square example: one-dimensional inside, zero outside.</span>
    </a>
    <div class="home-flow-arrow home-flow-arrow-out" aria-hidden="true"><span>ask</span><b>→</b></div>
    <div class="home-flow-outputs">
      <p class="home-flow-label">What next?</p>
      <a class="home-flow-item" href="topics/algebra.html"><strong>Algebra &amp; transport <span aria-hidden="true">↗</span></strong><span>Construct and relate modules</span></a>
      <a class="home-flow-item" href="topics/invariants.html"><strong>Invariants &amp; comparisons <span aria-hidden="true">↗</span></strong><span>Ask what a summary preserves</span></a>
      <a class="home-flow-item" href="guides/visualization.html"><strong>Features &amp; figures <span aria-hidden="true">↗</span></strong><span>Measure, inspect, and share</span></a>
    </div>
  </div>
  <figcaption id="home-workflow-caption">The finite model keeps both spaces and maps, with an assignment from the original parameters.
    In the square example, following <i>q</i> recovers its one-dimensional space ℚ.
    The <a href="tutorials/inspect_encoding.html">square lesson</a> works out how to recover maps as well.</figcaption>
</figure>
<div class="home-perspectives">
<section class="home-ambition" aria-labelledby="one-place-to-follow-the-question">
```

## One place to follow the question

Our ambition is an **all-in-one environment for multiparameter persistence**:
from preparing data and constructing a filtration to manipulating modules,
computing invariants, building numerical features, and visualizing the result.
We want those stages to work together, so that a new question can build on the
mathematical object already in hand.

The project grows out of **Ezra Miller's theory of modules over posets**.
[Finite encodings](finite_encodings.md) make its organizing idea concrete:
under the encoding's hypotheses, the finite model recovers the represented
module's spaces and maps. The same retained object connects mathematical
questions to practical computations.

```@raw html
</section>
<section class="home-evidence" aria-labelledby="performance-you-can-examine">
```

## Performance you can examine

We aim to make that breadth competitive with, and where possible faster than,
specialized suites. Our extensive [benchmark reports](../benchmarks/index.md)
put this ambition to an empirical test: matched mathematical requests,
independently checked answers, timings, memory measurements, and downloadable
results.

The [QPA comparison](../benchmarks/qpa.md) shows a substantial advantage in
the matched finite-module algebra tasks. The [PHAT comparison](../benchmarks/phat.md)
finds a smaller aggregate advantage for complete ordinary F₂ barcodes, with
individual cases favoring each tool. Each report states its inputs, versions,
timing conditions, and limits, so you can judge what the evidence means for
your work.

For the algorithms behind the results, read the
[implementation accounts](../implementation/index.md).

```@raw html
</section>
</div>
<section class="home-spectrum" aria-labelledby="from-theory-to-application">
```

## From theory to application

The documentation moves from understanding the mathematics, through exploring
the library, to completing a particular task. These are independent entrances:
start with the kind of answer you need.

```@raw html
<nav class="home-explore" aria-label="Choose a kind of treatment">
  <a href="collections/mathematics.html"><strong>Mathematics <span aria-hidden="true">→</span></strong><span>Understand the objects and why the constructions matter.</span></a>
  <a href="collections/using.html"><strong>Using TamerOp <span aria-hidden="true">→</span></strong><span>Explore capabilities, compare choices, and combine operations.</span></a>
  <a href="collections/recipes.html"><strong>Task recipes <span aria-hidden="true">→</span></strong><span>Follow code and focused explanations to a defined result.</span></a>
</nav>
<p class="home-finding">Find a route through the ideas in the <a href="reading_map.html">learning map</a>,
  browse a subject in the <a href="topic_map.html">topic map</a>, or look up an operation in
  <a href="collections/api.html">API reference</a>.</p>
</section>
</div>
```
