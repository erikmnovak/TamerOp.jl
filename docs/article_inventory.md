# Maintaining the article inventory

The [article catalog](article_inventory.toml) is the single inventory of existing
and planned documentation. It records the question each article owns, its
intended editorial type, and its topic relationships. Use it to place a new
article or plan a gradual revision of an existing one. The
[writing guide](writing.md) explains the editorial boundaries and the principles
for the learning map and topic atlas.

The [documentation programme](documentation_plan.md) develops the whole catalog
vertically within its five main article families and horizontally by subject.
Consult it before adding a page: it records intended groupings, shared examples,
the complete mathematical map design, and boundaries for future optimization
writing. The catalog remains authoritative for each article's scope and source.

An inventory entry is an article, not a function, source-code module, figure,
website route, or development task. A canonical notebook has one record: its
website lesson and executed download are generated presentations of that same
work. A guide covering several API families also has one record. Supporting
indexes, authored manifests, and project records use the separate `resource`
type because they need an owner without being forced into a prose genre.

## Read a record

| Field | Meaning |
| :--- | :--- |
| `id` | Stable article identity. Descriptive names are usual; B01–B72 preserve the implementation-account identities, and C01–C21 preserve the comparison identities. Renaming a title or moving its source does not change this ID. |
| `title` | Existing heading or working title. A planned title can change as its reader question becomes sharper. |
| `type` | Primary editorial purpose: `lesson`, `library_guide`, `recipe`, `api_reference`, `implementation_account`, `benchmark_report`, or `contributor_guide`. `resource` covers supporting indexes and records. |
| `topics` | Controlled subject IDs from the catalog's `[topics]` table. The first is the home cluster; further values are cross-topic membership. These are associations, not prerequisites. |
| `question` | The reader question this article is responsible for answering. Use it to detect overlap and keep a coherent scope. |
| `state` | `existing` means a canonical authored source exists; `planned` means it does not. This is **not** a publication, review, execution, mathematical-correctness, or software-implementation status. |
| `source` | Repository-relative path to the canonical existing source. It must exist, occur in only one record, and be absent on planned records. |
| `planned_source` | Optional destination already specified by a concrete plan. An absent value means no file location has been chosen. Do not create empty pages to fill it. |
| `scope` | Optional boundaries, substantive content, and qualifications. Conditional algorithm accounts and proposed comparisons must say what needs to be established before results can be written. |
| `plan_ids` | Related planning identifiers, such as B27, C18, or an A-item. They provide traceability, not a complete dependency graph or a claim that the underlying work is done. The record remains intelligible without local planning files. |
| `related` | Other article IDs worth consulting. These links express useful relationships; they do not define reading order, and need not be reciprocal. |
| `migration` | Editorial action: `retain`, `review`, `split`, or `author`. `retain` preserves the article's identity and purpose; it does not certify that every sentence is already current. |
| `migration_note` | Optional explanation of the proposed editorial change and the boundary it should respect. It is an authoring instruction, not text to place in a lesson. |

Types describe the intended home of existing material. A file classified as a
library guide can still contain algorithm detail that its migration note asks us
to relocate. Assigning a type is not a claim that the migration has happened.
Likewise, a planned implementation account can concern mature code, partly
implemented scope, or a conditional future algorithm. Check the code and its
contracts when drafting; the article catalog does not track feature completion.

The `lesson` family includes the mathematical world after encoding as well as
its construction: algebraic operations, changes of poset, invariants, summaries,
numerical features and interpretation of figures. The
[documentation programme](documentation_plan.md) describes the full set of
branches, with worked examples developed in the
[learning brief](learning_path.md#after-the-first-encoding-the-mathematical-curriculum).
`related` records remain associations; the proposed complete lesson relation
stays in the programme until authored lessons receive actual routes in
`reading_map.toml`.

## Keep metadata with its owner

The catalog joins documentation plans without replacing the manifests that
answer more specialized questions:

| Owner | Authoritative information |
| :--- | :--- |
| [Article catalog](article_inventory.toml) | Article identities, reader questions, editorial types, topic membership, canonical source presence, and migration intent. |
| [Documentation programme](documentation_plan.md) | Composition across and within the five main article families, shared examples, complete mathematical-map design, and staged migration/publication principles. No separate implementation or article-status ledger. |
| [Reading routes](reading_map.toml) | Curricular nodes, suggested continuations, optional detours, and lesson footer destinations. The arrows do not encode mandatory prerequisites; articles state those where needed. Topic membership does not create a reading-map arrow. |
| [Publication manifest](publication.toml) | Canonical-source-to-site routes, notebook execution, and generated page/download relationships. `existing` in the catalog does not mean published. |
| [Navigation display](navigation.toml) | Collection labels, grouping and display order. It refers to catalog types and topics; it does not duplicate article sources or decide learning continuations. |
| [API coverage](api_coverage.toml) | Binding and family coverage, reference targets, examples, method review, and reference-writing backlog. An article record for a target does not mean its methods are documented. |
| [API inventory guide](api_inventory.md) and generated runtime inventory | What bindings actually exist and how their identity and intended public exposure are reconciled. The generated symbol list is not an authored article. |
| [Learning-path brief](learning_path.md) | Worked mathematical examples, page-level teaching requirements, figure expectations, and acceptance evidence for the first route. |
| [Comparison-suite guide](benchmark_suites.md) | Experimental charters, comparison semantics, study scope and closure. A planned report supplies no measured result. |

Detailed algorithm and feature implementation status stays with its development
plan and evidence. Public documentation must stand without ignored local audit
trees. The B-series scopes are therefore recorded here in self-contained prose;
no catalog lookup, site build, or reader needs those local files. Existing local
planning lists can point to these article identities without becoming a second
independently maintained article inventory.

Keep authoring state and migration notes out of reader-facing lessons. The
generated topic map and collection landings show authored, published treatments;
they do not turn the backlog into empty cards. Source presence alone is
insufficient to decide whether an article belongs in public navigation: use
the publication manifest and editorial review for that decision.

`site_catalog.py` joins article metadata to publication routes before Documenter
runs. It generates collection and topic pages from those records, with a single
canonical destination for each article. After Documenter and the learning-route
footer pass, `site_shell.py` applies the shared collection navigation, separate
local heading outline and article metadata used by search. Generated pages and
HTML are build outputs; edit the responsible source or manifest instead.

The six collections are Mathematics, Using TamerOp, Task recipes, API reference,
Implementation and Benchmarks. Using TamerOp contains `library_guide` records;
Task recipes contains `recipe` records. Installation remains permanently
accessible above the collections, with both entrances resolving to the same
article. Contributor resources have a sidebar footer entrance. Topic pages
cross these collection boundaries without making their articles prerequisites
for one another. The first three collections distinguish understanding a
concept, exploring library capabilities and choices, and accomplishing a
defined task; they are independent entrances along a theory-to-application
spectrum. See
[site navigation maintenance](README.md#maintain-site-navigation) for review
checks.

## Maintain the collection gradually

### Plan documentation with a computational addition

Associate an implementation task with existing article identities before
creating new ones. A new input convention can extend its construction account
and usage guide; an internal optimization may need only an account revision
and performance evidence. A new mathematical question can justify a lesson,
and a new operational choice can justify a guide or bounded recipe. Do not
allocate a fixed number of pages per A-item. Preserve B-series identities and
keep research-dependent scopes conditional until there is an algorithm to
explain.

During parallel implementation, each feature owner supplies the mathematical
scope, example, reference-method changes and article IDs with the code. One
integration owner applies shared catalog, API-coverage and publication changes
against the combined source tree. Keep implementation verification, example
execution, source presence, editorial review and publication as distinct facts;
`state` continues to record source presence only. Schedule and completion
evidence belong to development records, not catalog fields or lesson prose.

### Add or revise an article

Before adding an article, look for the same reader question and review its
related records. Several views of a topic are useful when they serve different
purposes: a lesson explains why structure maps matter, a library guide explores
their queries, a recipe obtains a particular map and checks its result,
reference specifies the contract, and an implementation account explains how
maps are recovered. They should link to each other's authoritative
explanations instead of retelling the same treatment.

Start a planned record when there is a coherent reader question and bounded
scope. Do not commission a separate page for every option, feature request or
source file. Collection indexes can be planned before their individual case
studies are chosen, but a future case study gets its own record only when its
question and data are defined. A new algorithm's documentation plan must remain
conditional until there is an implemented method to explain.

Plan recipes by a concrete task with a clear starting input and stopping point.
Record the essential hypotheses, intended output and boundary with its related
guide in `scope`. Keep the recipe backlog in this catalog rather than creating
a parallel list. An existing `recipe` record can still need migration to a
runnable, focused procedure: record that work without implying it has already
happened. The [recipe-writing guidance](writing.md#write-recipes-around-an-attainable-result)
sets the intended style without requiring a separate article for every option.

When revising an existing hybrid, retain its useful explanations and correct
assumptions. Move a passage only when its new authoritative home is ready, then
replace repetition with a purposeful link. If one article becomes two, narrow
the original record and add one for the newly authored scope. If a planned guide
is fulfilled by reshaping an existing file, merge the overlapping plan records,
choose one stable identity, and update references instead of recording the file
twice. Do not move or archive a page merely because its current style differs
from the intended family.

Change `state` to `existing` and add `source` as soon as the canonical article is
authored; execution, publication and review work stays with its proper owner.
On publication, update the appropriate publication routes and review incoming
links. The build then derives the collection and topic listings; do not add
another source list to `make.jl` or `navigation.toml`. When a scope changes,
update its question, related articles and topics together. Review its search
type/topic metadata and its places in both collection and topic navigation.

The catalog includes canonical Markdown and notebooks in `docs/`, the repository
front door and contribution/release records, the browser-testing guide, and
authored navigation/coverage manifests. It excludes generated build pages,
notebook downloads, runtime symbol inventories, code docstrings, plotting assets,
benchmark data files, package environment manifests, and local audit evidence.
Benchmark bundle READMEs are resources because they explain the public data.
The catalog and this maintenance guide are included so they also have explicit
ownership.

## Check the catalog without Julia

From the repository root, this check parses the manifest and verifies identities,
source paths, relationships, types, and topic names. It also checks that every
planned API reference destination has exactly one article record, and that the
preserved B and C series are complete. These are structural checks, not evidence
that prose or software is correct.

```sh
python - <<'PY'
from pathlib import Path
import tomllib

root = Path.cwd()
catalog = tomllib.loads((root / "docs/article_inventory.toml").read_text())
articles = catalog["articles"]
ids = {item["id"] for item in articles}
assert len(ids) == len(articles), "duplicate article ID"
sources = [item["source"] for item in articles if "source" in item]
assert len(set(sources)) == len(sources), "duplicate canonical source"
for item in articles:
    assert item["type"] in catalog["types"], item["id"]
    assert item["topics"] and set(item["topics"]) <= set(catalog["topics"])
    assert item["state"] in {"existing", "planned"}, item["id"]
    assert item["migration"] in {"retain", "review", "split", "author"}
    assert set(item.get("related", [])) <= ids, item["id"]
    assert ("source" in item) == (item["state"] == "existing"), item["id"]
    for key in ("source", "planned_source"):
        if key in item:
            path = Path(item[key])
            assert not path.is_absolute() and ".." not in path.parts, item["id"]
    if "source" in item:
        assert (root / item["source"]).is_file(), item["source"]

coverage = tomllib.loads((root / "docs/api_coverage.toml").read_text())
targets = set(coverage["first_path"]["reference_pages"])
for family in coverage["families"]:
    targets.update(family["reference_pages"])
for target in targets:
    matches = [item for item in articles
               if target in (item.get("source"), item.get("planned_source"))]
    assert len(matches) == 1, (target, len(matches))
assert {f"B{i:02}" for i in range(1, 73)} <= ids
assert {f"C{i:02}" for i in range(1, 22)} <= ids
print(f"Validated {len(articles)} article and resource records.")
PY
```

Use a small query when reviewing one part of the backlog instead of maintaining
another status table. For example, the following lists existing and planned
library guides from the authoritative records:

```sh
python - <<'PY'
from pathlib import Path
import tomllib

catalog = tomllib.loads(Path("docs/article_inventory.toml").read_text())
for item in catalog["articles"]:
    if item["type"] == "library_guide":
        print(f"{item['id']}: {item['title']} ({item['state']})")
        print(f"  {item['question']}")
PY
```
