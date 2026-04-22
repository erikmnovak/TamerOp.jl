import csv
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "benchmark" / "thesis_macro_harness"


class ThesisMacroHarnessSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="thesis_macro_harness_", dir=HARNESS_DIR))
        cls.manifest = cls.tmpdir / "manifest.toml"
        cls.results = cls.tmpdir / "results.csv"
        cls.summary = cls.tmpdir / "summary.csv"
        cls.markdown = cls.tmpdir / "summary.md"
        cls.latex = cls.tmpdir / "summary.tex"
        cls.errors = cls.tmpdir / "errors.csv"

        cls._run([
            "julia", "--project=.", "benchmark/thesis_macro_harness/generate_catalog.jl",
            f"--profile=smoke", f"--out={cls.manifest}", "--fixtures_dir=fixtures",
        ])
        with cls.manifest.open("rb") as fh:
            cls.catalog = tomllib.load(fh)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    @classmethod
    def _run(cls, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, cwd=REPO_ROOT, check=True, text=True, capture_output=True)

    def _pick_source_case(self, source_kind: str) -> dict:
        for case in self.catalog["source_cases"]:
            if case["source_kind"] == source_kind:
                return case
        raise AssertionError(f"missing source case for {source_kind}")

    def _source_case_by_id(self, case_id: str) -> dict:
        for case in self.catalog["source_cases"]:
            if case["id"] == case_id:
                return case
        raise AssertionError(f"missing source case {case_id}")

    def _pick_job(self, *, family_case: str, invariant_kind: str, degree_label: str) -> dict:
        for job in self.catalog["jobs"]:
            if (
                job["family_case"] == family_case
                and job["invariant_kind"] == invariant_kind
                and job["degree_label"] == degree_label
            ):
                return job
        raise AssertionError(f"missing job {(family_case, invariant_kind, degree_label)}")

    def test_catalog_excludes_invalid_1param_mpp_jobs(self) -> None:
        self.assertEqual(self.catalog["harness"], "thesis_macro_v1")
        self.assertTrue(self.catalog["source_cases"])
        self.assertTrue(self.catalog["jobs"])
        bad = [
            job for job in self.catalog["jobs"]
            if job["family"] == "rips" and job["invariant_kind"] == "mpp_image"
        ]
        self.assertEqual(bad, [])
        self.assertIsNotNone(
            self._pick_job(
                family_case="pl_fringe_axis",
                invariant_kind="restricted_hilbert",
                degree_label="na",
            )
        )
        self.assertIsNotNone(
            self._pick_job(
                family_case="flange_friendly",
                invariant_kind="restricted_hilbert",
                degree_label="na",
            )
        )

    def test_fixture_generation_by_modality(self) -> None:
        source_cases = [
            self._source_case_by_id("pc_annulus2d_rips__n80"),
            self._source_case_by_id("image_checkerboard__n24"),
            self._source_case_by_id("pl_axis__n64"),
            self._source_case_by_id("flange_friendly__n64"),
        ]
        source_case_ids = ",".join(case["id"] for case in source_cases)
        self._run([
            "julia", "--project=.", "benchmark/thesis_macro_harness/generate_fixtures.jl",
            f"--manifest={self.manifest}", f"--source_case_ids={source_case_ids}", "--force=true",
        ])
        for case in source_cases:
            fixture_path = self.manifest.parent / case["fixture_relpath"]
            self.assertTrue(fixture_path.is_file(), fixture_path)

    def test_runner_missing_fixture_preflight(self) -> None:
        missing_case = self._source_case_by_id("pc_clusters8d_rips__n80")
        missing_job = next(
            job for job in self.catalog["jobs"]
            if job["source_case_id"] == missing_case["id"]
            and job["family_case"] == "rips"
            and job["invariant_kind"] == "euler_signed_measure"
            and job["degree_label"] == "na"
        )
        self.assertFalse((self.manifest.parent / missing_case["fixture_relpath"]).exists())
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self._run([
                "julia", "--project=.", "benchmark/thesis_macro_harness/run_tamer_macro.jl",
                f"--manifest={self.manifest}", f"--out={self.results}", "--profile=smoke",
                f"--job_ids={missing_job['id']}", "--fail_fast=true",
            ])
        self.assertIn("Missing required fixture files", ctx.exception.stderr)

    def test_degree_agnostic_rips_max_dim_default(self) -> None:
        out = self._run([
            "julia", "--project=.", "-e",
            (
                'include("benchmark/thesis_macro_harness/common.jl"); '
                'println(_max_dim_for_job(:rips, nothing)); '
                'println(_max_dim_for_job(:rips, 1));'
            ),
        ])
        lines = out.stdout.strip().splitlines()
        self.assertEqual(lines, ["1", "2"])

    def test_runner_smoke_and_summary_export(self) -> None:
        image_case = self._source_case_by_id("image_checkerboard__n24")
        self._run([
            "julia", "--project=.", "benchmark/thesis_macro_harness/generate_fixtures.jl",
            f"--manifest={self.manifest}", f"--source_case_ids={image_case['id']}", "--force=true",
        ])
        job = self._pick_job(
            family_case="lower_star",
            invariant_kind="euler_signed_measure",
            degree_label="na",
        )
        progress_path = Path(str(self.results) + ".progress")
        self._run([
            "julia", "--project=.", "benchmark/thesis_macro_harness/run_tamer_macro.jl",
            f"--manifest={self.manifest}", f"--out={self.results}", "--profile=smoke",
            f"--job_ids={job['id']}", "--fail_fast=true",
        ])

        with self.results.open("r", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 1)
        self.assertTrue(any(row["backend_label"] for row in rows))
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertTrue(progress_path.is_file())
        self.assertIn("status=completed", progress_path.read_text())
        self._run([
            sys.executable,
            "benchmark/thesis_macro_harness/summarize_results.py",
            f"--results={self.results}",
            f"--summary_out={self.summary}",
            f"--markdown_out={self.markdown}",
            f"--latex_out={self.latex}",
            f"--errors_out={self.errors}",
        ])
        self.assertTrue(self.summary.is_file())
        self.assertTrue(self.markdown.is_file())
        self.assertTrue(self.latex.is_file())
        with self.summary.open("r", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows)
        self.assertIn("family", rows[0])
        self.assertIn("cold_median_ms", rows[0])
        self.assertIn("| source_kind |", self.markdown.read_text())
        self.assertIn(r"\begin{tabular}", self.latex.read_text())


if __name__ == "__main__":
    unittest.main()
