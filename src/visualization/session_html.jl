# Portable, finite-state HTML uses the existing Cairo renderer. The browser only
# selects captured figures and records; it never evaluates Julia or data as code.

function _portable_svg(spec::VisualizationSpec; style=VisualStyle(), size=nothing)
    check_visual_spec(spec; throw=true)
    spec.kind === :linked_inspector && throw(ArgumentError(
        "Capture inspection_snapshot(session) before exporting a portable view."))
    get(spec.interaction, :requires_live_julia, false) && throw(ArgumentError(
        "Portable views require precomputed snapshots, not live Julia callbacks."))
    _check_visual_render_options(; style, size)
    backend = _resolve_visual_backend(:cairomakie; for_save=true)
    return mktempdir() do directory
        path = joinpath(directory, "view.svg")
        Base.invokelatest(_VISUAL_SAVERS[backend], path, spec; style, size)
        read(path, String)
    end
end

const _PORTABLE_VISUAL_CSS = raw"""
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#172b38;background:white}
*{box-sizing:border-box}body{margin:1.5rem auto;padding:0 1rem;max-width:1600px;line-height:1.5;overflow-wrap:anywhere}
h1{font-size:1.8rem;line-height:1.2}h2{font-size:1.25rem}p{max-width:85ch}
button,select,a{font:inherit}button,select{padding:.45rem .7rem;border:1px solid #78909c;border-radius:.25rem;background:white;color:inherit}
button:not(:disabled),select{cursor:pointer}button:disabled{color:#65747d;background:#f2f5f6}
button:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible,.figure-scroll:focus-visible{outline:3px solid #0072b2;outline-offset:3px}
nav{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;padding:1rem 0;border-block:1px solid #d8e0e4}
label{font-weight:600}select{max-width:100%;min-width:8rem;flex:1 1 16rem}nav .step{display:flex;gap:.5rem}
figure{margin:1.25rem 0}.figure-scroll{max-width:100%;overflow:auto;border:1px solid #d8e0e4;background:white}
.figure-scroll img{display:block;width:auto;max-width:none;height:auto}
figcaption{padding:.6rem 0}#view-status{min-height:1.5em;margin:.4rem 0;color:#405965}
.contract{border-left:4px solid #0072b2;background:#f2f7fa;padding:.65rem 1rem}
.downloads{display:flex;flex-wrap:wrap;gap:.65rem;margin:1rem 0}.downloads a{color:#005e88;padding:.35rem 0}
details{border-top:1px solid #d8e0e4;padding:.8rem 0}summary{cursor:pointer;font-weight:600}
pre{font: .9rem/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fa;padding:1rem;max-height:28rem;overflow:auto}
#saved-selection,#saved-context{margin:.7rem 0}.error{color:#a51b16}
@media(max-width:600px){body{margin:1rem auto;padding:0 .65rem}h1{font-size:1.45rem}nav{gap:.5rem}select{flex-basis:100%}.downloads button{width:100%}}
@media print{nav,.downloads{display:none}details{break-inside:avoid}figure{break-inside:avoid}.figure-scroll{overflow:visible}.figure-scroll img{max-width:100%}}
"""

const _PORTABLE_VISUAL_SCRIPT = raw"""
(() => {
  'use strict';
  const bundle = JSON.parse(document.getElementById('portable-visual-data').textContent);
  const byId = new Map(bundle.states.map((state, index) => [state.id, index]));
  const chooser = document.getElementById('saved-state');
  const image = document.getElementById('saved-figure');
  const status = document.getElementById('view-status');
  const caption = document.getElementById('saved-caption');
  const previous = document.getElementById('previous-state');
  const next = document.getElementById('next-state');
  let current = byId.get(bundle.initial);
  for (const state of bundle.states) {
    const option = document.createElement('option');
    option.value = state.id;
    option.textContent = state.label;
    chooser.append(option);
  }
  function requestedState() {
    if (!location.hash) return bundle.initial;
    try {
      return new URLSearchParams(location.hash.slice(1)).get('state');
    } catch (_) { return null; }
  }
  function display(id, updateHash = false) {
    const found = byId.has(id);
    current = found ? byId.get(id) : byId.get(bundle.initial);
    const state = bundle.states[current];
    chooser.value = state.id;
    image.src = state.image;
    image.alt = state.label + '. Precomputed selected figure. Its selection, context and plotted values are available below.';
    image.dataset.stateId = state.id;
    caption.textContent = state.label;
    document.getElementById('saved-selection').textContent = state.selection;
    document.getElementById('saved-context').textContent = state.context;
    document.getElementById('plotted-values').textContent = state.plotted_json;
    previous.disabled = current === 0;
    next.disabled = current === bundle.states.length - 1;
    status.textContent = (found ? '' : 'Unknown saved state; showing the initial view. ') +
      'Saved view ' + (current + 1) + ' of ' + bundle.states.length + ': ' + state.label;
    status.classList.toggle('error', !found);
    const hash = '#state=' + encodeURIComponent(state.id);
    document.getElementById('state-link').href = hash;
    const svg = document.getElementById('download-figure');
    svg.href = state.image;
    svg.download = 'view-' + (current + 1) + '.svg';
    if (updateHash && location.hash !== hash) location.hash = hash;
    document.documentElement.dataset.selectedState = state.id;
    document.documentElement.dataset.ready = 'true';
  }
  function download(text, name) {
    const url = URL.createObjectURL(new Blob([text], {type: 'application/json'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  chooser.addEventListener('change', () => display(chooser.value, true));
  previous.addEventListener('click', () => display(bundle.states[current - 1].id, true));
  next.addEventListener('click', () => display(bundle.states[current + 1].id, true));
  document.getElementById('download-session').addEventListener('click', () =>
    download(bundle.session_json, 'visual-session.json'));
  document.getElementById('download-values').addEventListener('click', () =>
    download(bundle.states[current].plotted_json, 'view-' + (current + 1) + '-values.json'));
  window.addEventListener('hashchange', () => display(requestedState()));
  display(requestedState());
})();
"""

function _write_portable_visual_html(path::AbstractString; title, caption, states, initial,
                                     session_json::AbstractString)
    isempty(states) && throw(ArgumentError("A portable session needs at least one saved view."))
    allunique(s.id for s in states) || throw(ArgumentError("Saved view identities must be unique."))
    any(s -> s.id == initial, states) || throw(ArgumentError("The initial saved view does not exist."))
    escaped(s) = replace(String(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
    records = [(; id=s.id, label=s.label,
        image="data:image/svg+xml;base64," * Base64.base64encode(s.svg),
        selection=s.selection, context=s.context, plotted_json=s.plotted_json) for s in states]
    # JSON remains inert text even for a title or label containing </script>.
    payload = replace(String(JSON3.write((; initial, states=records, session_json))),
        '<' => "\\u003c", '\u2028' => "\\u2028", '\u2029' => "\\u2029")
    saved_record = replace(String(session_json),
        '<' => "\\u003c", '\u2028' => "\\u2028", '\u2029' => "\\u2029")
    script_hash = Base64.base64encode(SHA.sha256(_PORTABLE_VISUAL_SCRIPT))
    style_hash = Base64.base64encode(SHA.sha256(_PORTABLE_VISUAL_CSS))
    policy = "default-src 'none'; script-src 'sha256-$script_hash'; style-src 'sha256-$style_hash'; " *
        "img-src data:; connect-src 'none'; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'"
    first_state = states[findfirst(s -> s.id == initial, states)]
    write(path, "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">" *
        "<meta http-equiv=\"Content-Security-Policy\" content=\"" * escaped(policy) * "\">" *
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>" *
        escaped(title) * "</title><style>" * _PORTABLE_VISUAL_CSS * "</style></head><body>" *
        "<header><h1>" * escaped(title) * "</h1><p>" * escaped(caption) * "</p></header>" *
        "<p class=\"contract\">This file contains precomputed views. Switching saved selections, " *
        "reading their records and downloading data work offline after Julia exits. " *
        "New mathematical queries and live inspector controls require Julia.</p>" *
        "<nav aria-label=\"Saved mathematical views\"><label for=\"saved-state\">Saved view</label>" *
        "<select id=\"saved-state\" aria-describedby=\"view-status\"></select>" *
        "<span class=\"step\"><button id=\"previous-state\" type=\"button\">Previous</button>" *
        "<button id=\"next-state\" type=\"button\">Next</button></span></nav>" *
        "<p id=\"view-status\" role=\"status\" aria-live=\"polite\"></p>" *
        "<figure><div class=\"figure-scroll\" tabindex=\"0\" role=\"region\" aria-label=\"Saved figure\"><img id=\"saved-figure\" src=\"data:image/svg+xml;base64," *
        Base64.base64encode(first_state.svg) * "\" alt=\"" * escaped(first_state.label) * "\"></div>" *
        "<figcaption id=\"saved-caption\">" * escaped(first_state.label) * "</figcaption></figure>" *
        "<div class=\"downloads\"><button id=\"download-session\" type=\"button\">Download session JSON</button>" *
        "<button id=\"download-values\" type=\"button\">Download selected values</button>" *
        "<a id=\"download-figure\">Download selected SVG</a><a id=\"state-link\">Link to this saved view</a></div>" *
        "<details open><summary>Saved selection</summary><pre id=\"saved-selection\"></pre></details>" *
        "<details><summary>Mathematical context and provenance</summary><pre id=\"saved-context\"></pre></details>" *
        "<details><summary>Machine-readable plotted values</summary><pre id=\"plotted-values\"></pre></details>" *
        "<noscript><p>The initial figure is available above. Enable JavaScript to switch saved views and download their records.</p></noscript>" *
        "<script id=\"tamerop-visual-session\" type=\"application/json\">" * saved_record * "</script>" *
        "<script id=\"portable-visual-data\" type=\"application/json\">" * payload * "</script>" *
        "<script>" * _PORTABLE_VISUAL_SCRIPT * "</script></body></html>\n")
    return String(path)
end
