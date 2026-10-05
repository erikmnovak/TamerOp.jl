#!/usr/bin/env python3
"""Execute canonical tutorials once, then build their Documenter pages/downloads."""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from urllib.parse import quote

import nbformat
from nbclient import NotebookClient
from nbconvert import MarkdownExporter
from jupyter_client.kernelspec import KernelSpecManager
from traitlets.config import Config

from reading_map import render_reading_map
from navigation import validate_continuations
from site_catalog import prepare_catalog

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
REPO = "https://github.com/erikmnovak/TamerOp.jl/blob/main/"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def heading_slug(title: str) -> str:
    return re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")


def documenter_markdown(text: str) -> str:
    """Translate notebook math outside fenced code to Documenter's math syntax."""
    parts = re.split(r"(^```[^\n]*\n.*?^```\s*$)", text, flags=re.M | re.S)
    heading_ids = {}
    def heading(match):
        level, title = match.groups()
        if "(@id " in title:
            return match[0]
        slug = heading_slug(title)
        count = heading_ids.get(slug, 0)
        heading_ids[slug] = count + 1
        anchor = slug if count == 0 else f"{slug}-{count}"
        return f"{level} [{title}](@id {anchor})"
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"^(#{1,6}) (.+)$", heading, parts[i], flags=re.M)
        part = re.sub(r"\$\$(.*?)\$\$", lambda m: "\n```math\n" + m[1].strip() + "\n```\n",
                      parts[i], flags=re.S)
        # Protect inline code while translating inline math.
        tokens = re.split(r"(`+[^`]*`+)", part)
        for j in range(0, len(tokens), 2):
            tokens[j] = re.sub(r"(?<!\\)\$([^$\n]+?)(?<!\\)\$", r"``\1``", tokens[j])
        parts[i] = "".join(tokens)
    return "".join(parts)


def figure_alt(text: str) -> str:
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("\n", " ")


def rewrite_links(text: str, source: Path, destination: Path | None,
                  pages: dict[Path, Path], stage: Path) -> str:
    """Resolve canonical-source links for the site or a portable download."""
    def replace(match: re.Match) -> str:
        url = match[1]
        if re.match(r"(?:[a-z]+:|#|/)", url):
            return match[0]
        name, sep, anchor = url.partition("#")
        target = (source.parent / name).resolve()
        if not target.is_file() and target not in pages:
            raise ValueError(f"Broken source link in {source.relative_to(ROOT)}: {url}")
        if destination is not None and target in pages:
            link = Path(os.path.relpath(pages[target], destination.parent)).as_posix()
        elif destination is not None and (target.is_relative_to(DOCS / "assets") or
                (target.is_relative_to(DOCS / "benchmarks") and target.suffix != ".md")):
            asset = stage / target.relative_to(DOCS)
            asset.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, asset)
            link = Path(os.path.relpath(asset, destination.parent)).as_posix()
        else:
            link = REPO + quote(target.relative_to(ROOT).as_posix())
        return "](" + link + (sep + anchor if sep else "") + ")"
    # Algebra such as k[U](q), and examples inside code, are not Markdown links.
    parts = re.split(r"(^```[^\n]*\n.*?^```[ \t]*$|\$\$.*?\$\$|(?<!\\)\$[^$\n]+(?<!\\)\$|`+[^`]+`+)",
                     text, flags=re.M | re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\]\(([^)\s]+)\)", replace, parts[i])
    return "".join(parts)


def validate_outputs(nb) -> int:
    """Require every cell and every declared teaching figure; errors never pass."""
    figures = 0
    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        if cell.execution_count is None:
            raise ValueError(f"Unexecuted cell: {cell.id}")
        if any(o.output_type == "error" for o in cell.outputs):
            raise ValueError(f"Failed cell: {cell.id}")
        images = [o for o in cell.outputs if any(k in o.get("data", {})
                  for k in ("image/png", "image/svg+xml"))]
        alt = cell.metadata.get("tamerop", {}).get("figure_alt", [])
        descriptions = [alt] if isinstance(alt, str) else alt
        if len(images) != len(descriptions) or any(not x.strip() for x in descriptions):
            raise ValueError(f"Missing figure or figure description in cell {cell.id}")
        for output, description in zip(images, descriptions):
            output.metadata["figure_alt"] = description
        figures += len(images)
    if not figures:
        raise ValueError("A published teaching notebook must contain static figures.")
    return figures


def export_lesson(nb, source: Path, page: Path, pages: dict, stage: Path) -> None:
    rendered = copy.deepcopy(nb)
    for cell in rendered.cells:
        if cell.cell_type == "markdown":
            cell.source = rewrite_links(cell.source, source, page, pages, stage)
    # Fold contiguous optional cells only in HTML. The portable notebook retains
    # ordinary cells and their saved results, with no HTML-specific authoring.
    cells, section, optional_ids = [], None, {}
    for cell in rendered.cells:
        next_section = cell.metadata.get("tamerop", {}).get("optional_section")
        if next_section is not None and (not isinstance(next_section, str) or not next_section.strip()):
            raise ValueError(f"optional_section must be a nonempty title in cell {cell.id}")
        if next_section != section:
            if section is not None:
                cells.append(nbformat.v4.new_markdown_cell("```@raw html\n</details>\n```"))
            if next_section is not None:
                anchor = ""
                heading = re.match(r"\A#{1,6} ([^\n]+)(?:\n|$)", cell.source) if cell.cell_type == "markdown" else None
                if heading and heading[1] == next_section:
                    # The disclosure already supplies this heading on the site.
                    # Retain its link target, and leave the notebook untouched.
                    slug = heading_slug(next_section)
                    count = optional_ids.get(slug, 0)
                    optional_ids[slug] = count + 1
                    anchor = ' id="' + html.escape(slug if count == 0 else f"{slug}-{count}") + '"'
                    cell.source = cell.source[heading.end():].lstrip()
                cells.append(nbformat.v4.new_markdown_cell(
                    '```@raw html\n<details class="optional-lesson"' + anchor + '>\n<summary>'
                    + html.escape(next_section) + '</summary>\n```'))
            section = next_section
        cells.append(cell)
    if section is not None:
        cells.append(nbformat.v4.new_markdown_cell("```@raw html\n</details>\n```"))
    rendered.cells = cells
    exporter = MarkdownExporter(
        template_file=str(DOCS / "build_scripts" / "documenter.md.j2"),
        filters={"documenter_markdown": documenter_markdown, "figure_alt": figure_alt},
        config=Config({"NbConvertBase": {
            "display_data_priority": ["image/png", "image/svg+xml", "text/plain"]}}),
    )
    body, resources = exporter.from_notebook_node(rendered, resources={
        "unique_key": source.stem, "output_files_dir": f"{source.stem}_files"})
    page.parent.mkdir(parents=True, exist_ok=True)
    for filename, data in resources.get("outputs", {}).items():
        output = page.parent / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    download = Path(os.path.relpath(stage / "downloads" / source.name, page.parent)).as_posix()
    record = Path(os.path.relpath(stage / "downloads" / "publication.json", page.parent)).as_posix()
    notice = (f"[Download the executed notebook]({download}) · "
              f"[Build and environment record]({record})\n\n"
              "All figures below are saved results; reading this page needs no Julia session. "
              "Select a figure to view it at full size.\n\n")
    # Keep the lesson's title first, followed by its generated download notice.
    title, rest = body.lstrip().split("\n", 1)
    page.write_text(f'```@meta\nEditURL = "{REPO}{source.relative_to(ROOT).as_posix()}"\n```\n\n'
                    + title + "\n\n" + notice + rest, encoding="utf-8")


def captured_notebook(source: Path, saved: Path, evidence: dict):
    """Reuse outputs only for the exact canonical lesson that produced them.

    Downloads have portable links, so restore their outputs onto the canonical
    cells before applying this build's publication routes.
    """
    if sha256(source) != evidence["sha256"] or sha256(saved) != evidence["download_sha256"]:
        raise ValueError(f"Changed notebook or captured download: {source.name}; execute a fresh publication.")
    authored = nbformat.read(source, as_version=4)
    executed = nbformat.read(saved, as_version=4)
    if len(authored.cells) != len(executed.cells):
        raise ValueError(f"Captured cell count differs: {source.name}")
    for cell, output in zip(authored.cells, executed.cells):
        if cell.cell_type != output.cell_type or cell.id != output.id:
            raise ValueError(f"Captured cell identity differs: {source.name}")
        if cell.cell_type == "code":
            if cell.source != output.source:
                raise ValueError(f"Captured code differs: {source.name}")
            cell.outputs = copy.deepcopy(output.outputs)
            cell.execution_count = output.execution_count
    if validate_outputs(authored) != evidence["figures"]:
        raise ValueError(f"Captured figures differ: {source.name}")
    return authored


def validate_capture(record: dict, previous: Path, config: dict,
                     source_hashes: dict, manifest_hash: str) -> None:
    """Fail closed when a formatting-only build would reuse stale execution."""
    if record["package_source_sha256"] != source_hashes or record["docs_manifest_sha256"] != manifest_hash:
        raise ValueError("Package or documentation environment changed; execute a fresh publication.")
    expected = {lesson["source"] for lesson in config["notebooks"]}
    recorded = [lesson["source"] for lesson in record["notebooks"]]
    if len(recorded) != len(set(recorded)) or set(recorded) != expected:
        raise ValueError("Notebook publication set changed; execute a fresh publication.")
    for name in ("Project.toml", "requirements.txt"):
        if sha256(DOCS / name) != sha256(previous / "environment" / name):
            raise ValueError(f"Execution environment changed ({name}); execute a fresh publication.")


def publish(julia: str, reuse_executed: bool = False) -> None:
    manifest = DOCS / "Manifest.toml"
    if not manifest.exists():
        raise RuntimeError("Instantiate docs/Project.toml first; see docs/README.md.")
    config = tomllib.loads((DOCS / "publication.toml").read_text())
    package_files = [ROOT / "Project.toml", *sorted((ROOT / "src").rglob("*.jl")),
                     *sorted((ROOT / "ext").rglob("*.jl"))]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in package_files}
    manifest_hash = sha256(manifest)
    build = DOCS / ".build"
    build.mkdir(exist_ok=True)
    previous = build / "src" / "downloads"
    capture = None
    if reuse_executed:
        capture = json.loads((previous / "publication.json").read_text())
        validate_capture(capture, previous, config, source_hashes, manifest_hash)
    env = {**os.environ, "JULIA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "JULIA_NUM_PRECOMPILE_TASKS": "1", "TAMEROP_DOCS_PYTHON": sys.executable}
    with tempfile.TemporaryDirectory(prefix="publication-", dir=build) as temporary:
        temporary = Path(temporary)
        stage = temporary / "src"
        shutil.copytree(DOCS / "src", stage)
        downloads = stage / "downloads"
        if reuse_executed:
            shutil.copytree(previous, downloads)
        else:
            downloads.mkdir()
        pages = {(DOCS / "src" / p.relative_to(stage)).resolve(): p
                 for p in stage.rglob("*.md")}
        pages.update({(DOCS / src).resolve(): stage / dest for src, dest in config["guides"].items()})
        pages.update({(DOCS / n["source"]).resolve(): stage / n["page"] for n in config["notebooks"]})
        authored = list(pages.items())
        pages.update({(DOCS / "src" / p.relative_to(stage)).resolve(): p for p in pages.values()})
        pages.update({(DOCS / "src" / "downloads" / Path(n["source"]).name).resolve():
                      downloads / Path(n["source"]).name for n in config["notebooks"]})
        for source, page in authored:
            if source.suffix != ".md":
                continue
            text = rewrite_links(source.read_text(), source, page, pages, stage)
            page.parent.mkdir(parents=True, exist_ok=True)
            page.write_text(f'```@meta\nEditURL = "{REPO}{source.relative_to(ROOT).as_posix()}"\n```\n\n'
                            + documenter_markdown(text))
        if "reading_map" in config:
            map_config = config["reading_map"]
            map_page = stage / map_config["page"]
            map_template = map_page.read_text()
            marker = "<!-- READING_MAP -->"
            if map_template.count(marker) != 1:
                raise ValueError("The reading-map page must contain exactly one READING_MAP marker")
            reading_config = tomllib.loads((DOCS / map_config["source"]).read_text())
            validate_continuations(reading_config, DOCS)
            map_html = render_reading_map(
                reading_config,
                {source: destination for source, destination in pages.items()
                 if source.is_relative_to(DOCS)}, stage, map_page, DOCS, REPO)
            map_page.write_text(map_template.replace(marker, f"```@raw html\n{map_html}\n```"))
        prepare_catalog(DOCS, stage)
        # Private kernel specification: no global Jupyter kernel is installed/changed.
        kernel_dir = temporary / "kernels" / "tamerop-docs"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [julia, "--startup-file=no", "--threads=1", f"--project={DOCS}",
                     "-i", "--color=no", "-e", "import IJulia; IJulia.run_kernel()", "{connection_file}"],
            "display_name": "TamerOp documentation", "language": "julia", "env": {
                k: env[k] for k in ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JULIA_NUM_PRECOMPILE_TASKS")}}))
        records = []
        for lesson in config["notebooks"]:
            source = DOCS / lesson["source"]
            notebook_hash = sha256(source)
            if reuse_executed:
                evidence = next(item for item in capture["notebooks"] if item["source"] == lesson["source"])
                nb = captured_notebook(source, downloads / source.name, evidence)
                export_lesson(nb, source, stage / lesson["page"], pages, stage)
                records.append(evidence)
                print(f"Reusing verified execution of {source.relative_to(ROOT)}", flush=True)
                continue
            nb = nbformat.read(source, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell.outputs = []
                    cell.execution_count = None
            work = temporary / "work" / source.stem
            work.mkdir(parents=True)
            print(f"Executing {source.relative_to(ROOT)} in a fresh Julia kernel", flush=True)
            started = time.monotonic()
            client = NotebookClient(nb, kernel_name="tamerop-docs",
                timeout=lesson["timeout"], startup_timeout=300, allow_errors=False,
                resources={"metadata": {"path": str(work)}},
                on_cell_execute=lambda cell, cell_index: print(f"  cell {cell_index + 1}", flush=True))
            client.create_kernel_manager().kernel_spec_manager = KernelSpecManager(
                kernel_dirs=[str(kernel_dir.parent)], ensure_native_kernel=False)
            try:
                client.execute()
                figures = validate_outputs(nb)
            except Exception:
                nbformat.write(nb, build / f"{source.stem}-failed.ipynb")
                raise
            elapsed = time.monotonic() - started
            if sha256(source) != notebook_hash:
                raise ValueError(f"Notebook changed during execution: {source.name}; rebuild it.")
            page = stage / lesson["page"]
            export_lesson(nb, source, page, pages, stage)
            portable = copy.deepcopy(nb)
            for cell in portable.cells:
                if cell.cell_type == "markdown":
                    cell.source = rewrite_links(cell.source, source, None, pages, stage)
            nbformat.write(portable, downloads / source.name)
            if (downloads / source.name).stat().st_size > 20 * 1024 * 1024:
                raise ValueError(f"Notebook exceeds the 20 MiB publication budget: {source.name}")
            exports = work / "tamerop_outputs"
            if exports.exists():
                shutil.copytree(exports, downloads / "figures" / source.stem)
            records.append({"source": lesson["source"], "sha256": notebook_hash,
                "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                "figures": figures, "seconds": round(elapsed, 3),
                "download_sha256": sha256(downloads / source.name)})
        environment = downloads / "environment"
        if not reuse_executed:
            environment.mkdir()
            for name in ("Project.toml", "Manifest.toml", "requirements.txt", "publication.toml"):
                shutil.copy2(DOCS / name, environment / name)
        if source_hashes != {p.relative_to(ROOT).as_posix(): sha256(p) for p in package_files} or sha256(manifest) != manifest_hash:
            raise ValueError("Package source or documentation environment changed during execution; rebuild.")
        record = capture if reuse_executed else {"notebooks": records,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
            "julia": subprocess.check_output([julia, "--startup-file=no", "--version"], env=env, text=True).strip(),
            "python_packages": {p: importlib.metadata.version(p) for p in
                ("nbclient", "nbconvert", "nbformat", "jupyter_client")},
            "package_source_sha256": source_hashes,
            "docs_manifest_sha256": manifest_hash}
        (downloads / "publication.json").write_text(json.dumps(record, indent=2) + "\n")
        target = build / "src"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(stage, target)
        for lesson in config["notebooks"]:
            (build / f"{Path(lesson['source']).stem}-failed.ipynb").unlink(missing_ok=True)
    print("Building Documenter HTML from the captured outputs", flush=True)
    subprocess.run([julia, "--startup-file=no", "--threads=1", f"--project={DOCS}",
                    str(DOCS / "make.jl")], cwd=ROOT, env=env, check=True)
    # Retain byte-identical data downloads, including the raw data dictionaries,
    # alongside their HTML pages so the published SHA256SUMS remain verifiable.
    for asset in (DOCS / "benchmarks").rglob("*"):
        if asset.is_file() and asset.parent != DOCS / "benchmarks":
            target = DOCS / "build" / asset.relative_to(DOCS)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, target)
    print(f"Open {DOCS / 'build' / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default=shutil.which("julia"), help="Julia executable (default: PATH)")
    parser.add_argument("--reuse-executed", action="store_true",
                        help="Rebuild prose/navigation using verified unchanged notebook executions")
    args = parser.parse_args()
    if not args.julia:
        parser.error("Julia is not on PATH; supply --julia /path/to/julia")
    publish(args.julia, reuse_executed=args.reuse_executed)
