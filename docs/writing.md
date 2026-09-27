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

Do not infer encoding-independent Ext or Tor from recovery of an ambient
module. State the finite category of a derived computation and link to the
[comparison hypotheses](math_categories.md) when relevant.

Use examples with known answers. Verify maps as well as dimensions when
the claim concerns a module or a comparison. A successful computation supports
a worked example; a general theorem needs a reference or derivation.

Figures should advance the explanation. Reuse labels across a parameter
picture, its finite poset, and the corresponding spaces and maps. Explain in
the caption what the reader should notice. Include the source of generated
figures and the conditions under which the displayed result is computed.

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
