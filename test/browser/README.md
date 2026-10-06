# Live inspector browser checks

These checks follow a parameter selection into the finite model: its space,
structure map, retained presentation, and restriction to a line. They also
exercise ordinary persistence intervals and retained source representatives.
Chromium connects to a running local Bonito server. Browser controls and real
canvas pointer events must reach Julia, update the mathematical selection, and return the
corresponding readout and drawing to the browser. Screenshots and failure traces
provide evidence of the rendered result.

This is an optional, focused browser suite. It does not run the package suite or
certify notebook frontends, Firefox, or other browsers. The broader
[A41 acceptance checklist](../../docs/testing.md#a41-interval-semantics-and-retained-representatives)
also includes fixtures and visual checks beyond this harness. Installing the
tools alone does not establish that any browser check has passed.

## Set up the dependencies

Use Julia 1.12 and Node.js 22 or another version supported by the pinned
Playwright packages. The lockfile pins `@playwright/test` and its `playwright`
CLI to 1.63.0, and the optional browser MCP server to `@playwright/mcp` 0.0.83.

From the repository root, prepare the optional Julia environment if needed:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --startup-file=no --project=test/browser \
  -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate(); Pkg.precompile()'
```

This develops the current checkout into the browser environment and uses one
precompilation task. An existing environment containing this checkout,
WGLMakie, and JSON3 can be used instead through `TAMEROP_JULIA_PROJECT`.

Install the JavaScript dependencies and both Chromium builds:

```sh
cd test/browser
npm ci
npx playwright install chromium
```

The ordinary interaction scenarios use Chromium's headless shell. The browser
zoom scenario needs the full Chromium build, because it loads the local test
extension described below. The installation command supplies both builds.

The Chromium sandbox remains enabled. Use a host that supports sandboxed
Chromium and its system libraries; the configuration does not disable the
sandbox to work around an unsupported host.

## Run the checks

From `test/browser`:

```sh
npm test
```

[playwright.config.mjs](playwright.config.mjs) starts [serve.jl](serve.jl), waits
for the interval server, runs one worker, and stops the Julia process it started.
Julia stays alive throughout the checks. Static exported HTML cannot substitute
for this connection.

Normal teardown uses [stop-server.mjs](stop-server.mjs): it creates the fixture's
stop file, allowing Julia to close its Bonito servers and inspection sessions,
then waits for the process identified by `/health` to exit. The process-existence
check sends no signal. Playwright retains a signal fallback if normal teardown
fails. When `TAMEROP_BROWSER_REUSE_SERVER=1`, teardown leaves the manually
started server running.

The fixture families use two loopback addresses. The additional square routes are
constructed when first requested and have independent mathematical sessions:

| Address | Fixture and checks | Style |
| --- | --- | --- |
| `http://127.0.0.1:8848` | Ordinary intervals: chart picking, duplicate members, retained cycles and bounding chains, linked tabs and closure | Accessible; configurable font size |
| `http://127.0.0.1:8849` | Whole-line band encoding: finite and infinite endpoints, window censoring, draft/apply controls and linked charts | Accessible; configurable font size |
| `http://127.0.0.1:8849/squares` | Two closed squares: exact spaces and boundaries, rank-one maps and their zero composite, incomparable parameters, active presentation blocks and image bases, hover and lifecycle | Accessible, 18px |
| `http://127.0.0.1:8849/squares-slices` | The same square module restricted to lines: closed intervals, tangent singletons, empty restrictions, invalid-input recovery and integration with stalk/map selection | Accessible, 18px |
| `http://127.0.0.1:8849/squares-grayscale` | The square module with larger text: source/target distinctions, keyboard focus, horizontal chart scrolling and real 150%/200% browser zoom | Grayscale, 24px |

The square module is the direct sum supported on `[0,2]^2` and `[1,3]^2`, over
the rational numbers. Its diagonal restriction has the closed intervals
`[0,2]` and `[1,3]`. These independent expected values connect the historical
manual checks to the maintained browser suite.

Each page creates a fresh viewer. Pages for the same fixture share its inspection
session, which allows linked selection and viewer-disconnection checks. Closing
the shared inspector closes that fixture session; restart the server before
another run that needs it.

The following environment variables customize the run:

| Variable | Default and purpose |
| --- | --- |
| `TAMEROP_JULIA` | `julia`; Julia executable path |
| `TAMEROP_JULIA_PROJECT` | This directory; Julia project containing the fixture dependencies |
| `TAMEROP_BROWSER_INTERVAL_PORT` | `8848`; ordinary interval server port |
| `TAMEROP_BROWSER_SLICE_PORT` | `8849`; linked slice server port |
| `TAMEROP_BROWSER_REUSE_SERVER` | Set to `1` to use an already running fixture server |
| `TAMEROP_BROWSER_OUTPUT` | `test-results` in this directory; test artifact directory |
| `TAMEROP_BROWSER_FONTSIZE` | `18`; positive font size for the ordinary and band fixtures; the three square routes retain the styles above |
| `TAMEROP_BROWSER_STOP_FILE` | `.stop-server` in this directory; shutdown request shared by the runner and Julia fixture |

The Julia subprocess inherits `JULIA_DEPOT_PATH`. When reusing a server, start
the fixtures from this version of `serve.jl`, use matching ports, and stop the
server yourself afterward. Set the font size when starting that server; a test
run cannot change the style of an already running fixture. A server can be
started from the repository root:

```sh
JULIA_NUM_PRECOMPILE_TASKS=1 julia --startup-file=no --threads=1 \
  --project=test/browser test/browser/serve.jl
```

In another terminal, run `TAMEROP_BROWSER_REUSE_SERVER=1 npm test` from this
directory. To stop the server normally, create `test/browser/.stop-server` from
the repository root. `TAMEROP_BROWSER_STOP_FILE` can specify another stop file;
use the same setting for the server and runner. Interrupting Julia remains a
fallback when a normal stop cannot complete.

For a visible browser, use `npm run test:headed`. To inspect a
completed run, use `npx playwright show-report`. Failure artifacts include
screenshots and traces under `test-results`; the HTML report is under
`playwright-report`. These directories, `node_modules`, the local Julia
`Manifest.toml`, and the stop file are ignored by Git.
Traces retain actions and DOM snapshots without continuously recording WebGL
frames. Explicit checkpoints and failures still save screenshots, keeping the
evidence useful without duplicating a large sequence of plot images.

## Browser zoom and horizontal access

At a narrow width, the plots retain their drawing dimensions inside scrollable
panels. The tests focus each panel and use the left/right arrow keys to reach
both ends. They also click the barcode and the right-hand diagram after
scrolling, checking that Julia receives the intended selection. The page itself
must fit the viewport, and scrolling must preserve the mathematical query.

[style-zoom.spec.mjs](style-zoom.spec.mjs) launches full Chromium with a fresh,
temporary profile and the repository's [zoom extension](zoom-extension/manifest.json).
The extension has loopback-only host permissions, no content scripts, and is
loaded only into this test browser. It uses Chrome's
[`tabs.setZoom`](https://developer.chrome.com/docs/extensions/reference/api/tabs#method-setZoom)
API to apply 150% and 200% page zoom, then reads the resulting factor back.
The test also checks the device-pixel-ratio change and that the visual viewport
has not been pinch-zoomed. This tests browser page zoom separately from larger
plot text or a smaller viewport. Playwright's
[extension guide](https://playwright.dev/docs/chrome-extensions) explains the
persistent-context and Chromium setup used here.

At each zoom level, the scenario checks visible keyboard focus, reaches both
charts by scrolling, selects intervals through real pointer events, and checks
that the previous incomparable-parameter query and exact endpoints are retained.
The temporary profile closes with the scenario; the user's browser profile and
zoom preferences are not used.

Zoom evidence includes separate viewport captures of the controls and both
scrolled chart ends. The helper requests Chromium's visible surface directly
through `Page.captureScreenshot`, with no document-coordinate clip and with
capture beyond the viewport disabled. This avoids blank or incorrectly cropped
captures seen after deep scrolling at real page zoom. Ordinary checkpoints
retain Playwright's full-page screenshots. Each image has a matching read-only
Julia-state attachment; the direct capture changes neither the page nor its
mathematical selection. Review the resulting images before claiming visual
acceptance of a run.

## Optional browser MCP

An existing Node installation and Julia environment can be reused through the
environment variables above. Keep the selected Node executable on `PATH`; if
the browser installation uses a custom cache, set `PLAYWRIGHT_BROWSERS_PATH` consistently when installing
and running Chromium. Machine-specific paths belong in the local run record.

For interactive agent verification, register the pinned `@playwright/mcp` server
using the [Codex MCP configuration](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).
Use `--headless --isolated --sandbox`, and provide `--executable-path` when using
an existing Chromium headless shell. The MCP package and test runner may require
different browser builds; record the executable actually used. Select
**Restart extension** to load newly configured tools. The automated `npm test`
command runs independently of that MCP connection.

## Interpreting the evidence

The fixture's hidden `julia-session-state` readout is observation only: it exposes
selection, exact interval records, module and presentation matrices, displayed
highlights, hover text, viewer counts, and canvas target coordinates. Matrix
shapes are preserved so an empty `1 x 0` image basis can be distinguished from
a `1 x 1` active zero block. The readout does not invoke selection callbacks.
Browser tests use the inspector controls and actual mouse events, then compare their effects with this
readout and visible UI. Canvas coordinates are taken from the rendered axes;
pointer delivery is checked before clicking because WGLMakie throttles movement
events.

On 3 October 2026, all five scenarios passed in one fresh-server Chromium 153
run (24.4 minutes). The command exited with status zero and no termination
signal. Julia recorded normal shutdown; the fixture PID was gone, both listener
ports were closed, and the stop marker was absent. The run used 24px for the
ordinary/band fixtures, 18px for the two historical square fixtures and 24px
grayscale for the zoom fixture, with 640px/1440px viewport checks.
All six direct zoom viewport captures—controls and both chart ends at 150% and
200%—passed visual review.

The previously crowded finite/infinity x-tick labels now rotate when needed,
retaining every tick value and diagram point. The actual band chart canvas and
narrow-right browser capture passed visual review. The separate focused native
diagram run passed 958/958 assertions, including measured label separation in
Cairo and WGL at fonts 18/24 and narrow widths. Very small standalone figures
can still need more height for explanatory headings. Barcode infinity text
remains inside its axis and above selection strokes.

The 2 October results remain earlier evidence: two original scenarios, then
four main-run cases plus a focused zoom rerun. That day's 1,228/1,228 native
owner-file result covered the barcode correction. Native counts overlap and
must not be added to each other or to the five browser scenarios.

Review screenshots for readable labels, correct endpoint decorations, and
layout at the tested widths. Larger text, device pixel density, and actual
browser zoom are different conditions; evidence for one does not certify the
others. Record the source revision, browser version, command, and result when
reporting acceptance. Keep notebook integration, other browsers, and untested
items from the A41 checklist explicitly separate.

## Anchored rank sections

`rank-sections.spec.mjs` uses an independent two-square session at `/rank-sections` to check both
anchored directions, the selected matrices, zero composites and incomparable
points within one fiber. It also checks real pointer selection, hover without
new algebra, input recovery, narrow layout, view switching, reset and closure.
The section canvas is additional to the navigation canvas; both remain linked
to the same Julia selection.


For the distance explorer, `matching.spec.mjs` exercises `/matching` and
`/matching-slices` on the slice server. The first fixture has repeated finite
intervals, a diagonal assignment and an essential match. The second has two
endpoint samples of cost one and an exact maximizing slice of cost `5/4`.
The exact button must change the scope label; sampled and selected values do
not certify an optimum. Pair selection and hover must leave the mathematical
query count unchanged. Test telemetry only observes state and plot coordinates.
The matching scenario opens an independent linked viewer, verifies shared
selection and disconnect cleanup, and uses keyboard arrows to reach both ends
of the charts at a narrow viewport. Bonito's reconnect grace is allowed before
requiring the disconnected viewer to be disposed.
