"""Command line interface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

from paper_agent.agents.draft_report import DraftReportAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.agents.llm_self_review import LLMSelfReviewAgent
from paper_agent.agents.submission_package_validator import SubmissionPackageValidatorAgent
from paper_agent.agents.submission_readiness import SubmissionReadinessAgent
from paper_agent.config import LLMConfig, load_llm_config
from paper_agent.export import zip_latex_project
from paper_agent.experiment_artifact_consistency import assess_experiment_artifact_consistency
from paper_agent.experiment_contract import experiment_results_template, validate_experiment_contract
from paper_agent.experiment_evidence import classify_experiment_evidence
from paper_agent.experiment_provenance import assess_experiment_provenance
from paper_agent.experiment_quality import assess_experiment_quality, tcga_experiment_quality_kwargs
from paper_agent.llm import ChatMessage, LLMClient, LLMError
from paper_agent.state import DraftSections, ExperimentSummary, PaperRequest
from paper_agent.tcga_artifacts import write_tcga_artifact_exports_from_rows
from paper_agent.workflow import PaperWorkflow


TCGA_READINESS_CONTRACT_SCHEMA_VERSION = "tcga-readiness-contract/v1"
TCGA_READINESS_CONTRACT_CATEGORIES = (
    "venue_network",
    "baseline_pdf",
    "code_path",
    "result_artifacts",
    "experiment_results",
    "llm",
    "latex",
    "pipeline_stage",
)
TCGA_READINESS_REQUIREMENT_STATUSES = (
    "pass",
    "warn",
    "fail",
    "disabled",
    "ready_to_fill",
    "ready_to_generate",
)


def _add_experiment_contract_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require ablation evidence for a complete experiment contract.",
    )
    parser.add_argument(
        "--require-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require sensitivity-analysis evidence for a complete experiment contract.",
    )
    parser.add_argument(
        "--require-statistical-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require statistical-test evidence for a complete experiment contract.",
    )


def _add_experiment_quality_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-dataset",
        action="append",
        default=[],
        help="Expected dataset/cohort name for result-file quality checks; can be repeated.",
    )
    parser.add_argument(
        "--expected-metric",
        action="append",
        default=[],
        help="Expected metric name for result-file quality checks; can be repeated.",
    )
    parser.add_argument(
        "--expected-method",
        default="",
        help="Expected proposed-method row name for result-file quality checks.",
    )
    parser.add_argument(
        "--expected-baseline",
        default="",
        help="Expected baseline row name for result-file quality checks.",
    )


def _add_experiment_provenance_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help=(
            "Require a result provenance table whose local artifact paths resolve. "
            "Use this for submission-grade result files."
        ),
    )


def _add_experiment_artifact_consistency_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-artifact-consistency",
        action="store_true",
        help=(
            "Require checkable local CSV provenance artifacts whose values match the parsed paper result tables. "
            "CSV columns should include method,dataset,metric,value."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="paper-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a deterministic demo draft.")
    demo.add_argument("--output", default="outputs/demo", help="Output directory for markdown.")
    demo.add_argument("--zip", default="", help="Optional path for an Overleaf-ready LaTeX zip.")
    demo.add_argument("--summary", default="", help="Optional path for a JSON run summary.")
    demo.add_argument(
        "--acceptance-report",
        default="",
        help="Optional path for a Markdown acceptance report.",
    )
    demo.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    demo.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    demo.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass for this run.",
    )
    draft = sub.add_parser("draft", help="Draft a paper from local research materials.")
    draft.add_argument("--project-name", required=True)
    draft.add_argument("--target-venue", required=True)
    draft.add_argument("--baseline", required=True, help="Baseline PDF path or directory containing a PDF.")
    draft.add_argument("--code-path", required=True)
    draft.add_argument("--experiment-results", required=True, help="Markdown/CSV/text experiment results file.")
    draft.add_argument("--keyword", action="append", default=[], help="Keyword; can be repeated.")
    draft.add_argument("--output", default="", help="Optional path for generated Markdown copy.")
    draft.add_argument("--zip", default="", help="Optional path for an Overleaf-ready LaTeX zip.")
    draft.add_argument("--summary", default="", help="Optional path for a JSON run summary.")
    draft.add_argument(
        "--acceptance-report",
        default="",
        help=(
            "Optional path for a Markdown acceptance report. Defaults next to "
            "--summary, or next to --output when no summary path is provided."
        ),
    )
    draft.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    draft.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    draft_network = draft.add_mutually_exclusive_group()
    draft_network.add_argument(
        "--online",
        action="store_true",
        help="Allow template/reference/related-work network calls for this draft run.",
    )
    draft_network.add_argument(
        "--offline",
        action="store_true",
        help="Disable template/reference/related-work network calls for this draft run.",
    )
    draft_llm = draft.add_mutually_exclusive_group()
    draft_llm.add_argument(
        "--allow-llm",
        action="store_true",
        help="Allow configured LLM section calls even if PAPER_AGENT_DISABLE_LLM was set.",
    )
    draft_llm.add_argument(
        "--disable-llm",
        action="store_true",
        help="Force deterministic local section drafting for this run.",
    )
    draft.add_argument(
        "--compile-latex",
        action="store_true",
        help="Run the local LaTeX compiler during submission validation.",
    )
    draft.add_argument(
        "--min-llm-sections",
        type=int,
        default=0,
        help="Fail the draft command unless at least this many sections are written by the LLM.",
    )
    draft.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass for this run.",
    )
    draft.add_argument(
        "--strict-results",
        action="store_true",
        help="Fail before generation unless the experiment result file is real and contract-complete.",
    )
    _add_experiment_contract_options(draft)
    _add_experiment_provenance_options(draft)
    _add_experiment_artifact_consistency_options(draft)
    sample = sub.add_parser(
        "sample-hyper-protosurv",
        help="Run the local Hyper-ProtoSurv example and write showcase artifacts.",
    )
    sample.add_argument(
        "--project-name",
        default="",
        help="Project name for generated LaTeX artifacts. Defaults to the output directory name.",
    )
    sample.add_argument("--example-root", default=r"D:\code\agent\example")
    sample.add_argument("--output-dir", default="outputs/hyper-protosurv-sample")
    sample.add_argument("--zip", default="outputs/hyper-protosurv-sample-overleaf.zip")
    sample.add_argument(
        "--acceptance-report",
        default="",
        help="Optional path for a Markdown acceptance report. Defaults to output-dir/ACCEPTANCE_REPORT.md.",
    )
    sample.add_argument(
        "--experiment-results",
        default="",
        help=(
            "Optional experiment table file. If omitted, the sample builds a TCGA cohort-data "
            "summary from code_path/dataset_csv/*.csv without fabricating performance scores."
        ),
    )
    sample.add_argument(
        "--online",
        action="store_true",
        help="Allow template/reference network calls. The default sample run is offline.",
    )
    sample.add_argument(
        "--compile-latex",
        action="store_true",
        help="Run the local LaTeX compiler during submission validation.",
    )
    sample.add_argument(
        "--allow-llm",
        action="store_true",
        help="Allow configured LLM calls. The default sample run is deterministic and local.",
    )
    sample.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass even when --allow-llm is used.",
    )
    sample.add_argument(
        "--strict-results",
        action="store_true",
        help="Fail before generation unless the experiment input is real and contract-complete.",
    )
    _add_experiment_contract_options(sample)
    _add_experiment_provenance_options(sample)
    _add_experiment_artifact_consistency_options(sample)
    tcga_doctor = sub.add_parser(
        "tcga-doctor",
        help="Check local Hyper-ProtoSurv TCGA inputs before running paper generation.",
    )
    tcga_doctor.add_argument("--example-root", default=r"D:\code\agent\example")
    tcga_doctor.add_argument(
        "--experiment-results",
        default="",
        help="TCGA result file. Defaults to example-root/results/tcga_results.md.",
    )
    tcga_doctor.add_argument(
        "--submission-grade",
        action="store_true",
        help="Also require provenance, artifact consistency, LaTeX compiler, and LLM configuration.",
    )
    tcga_doctor.add_argument(
        "--write-template",
        action="store_true",
        help="Write a missing TCGA result template at the expected experiment-results path.",
    )
    tcga_doctor.add_argument(
        "--live-llm",
        action="store_true",
        help="Run a live LLM provider preflight in addition to static configuration checks.",
    )
    tcga_doctor.add_argument("--summary", default="", help="Optional JSON doctor summary path.")
    _add_experiment_contract_options(tcga_doctor)
    _add_experiment_quality_options(tcga_doctor)
    _add_experiment_provenance_options(tcga_doctor)
    _add_experiment_artifact_consistency_options(tcga_doctor)
    tcga_results = sub.add_parser(
        "tcga-results-from-artifacts",
        help="Build a TCGA result Markdown file from CSV artifacts and provenance hashes.",
    )
    tcga_results.add_argument("--example-root", default=r"D:\code\agent\example")
    tcga_results.add_argument(
        "--artifacts-dir",
        default="",
        help="Directory to scan recursively for TCGA result CSV artifacts when explicit CSV paths are omitted.",
    )
    tcga_results.add_argument("--main-csv", default="", help="Main result CSV; wide or long format.")
    tcga_results.add_argument("--ablation-csv", default="", help="Optional ablation CSV.")
    tcga_results.add_argument("--sensitivity-csv", default="", help="Optional sensitivity CSV.")
    tcga_results.add_argument("--stats-csv", default="", help="Optional statistical-test CSV.")
    tcga_results.add_argument("--output", default="", help="Output Markdown path. Defaults to example-root/results/tcga_results.md.")
    tcga_results.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_results.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_results.add_argument("--metric", default="C-index")
    tcga_results.add_argument("--dataset", action="append", default=[], help="Dataset/cohort name; can be repeated.")
    tcga_results.add_argument(
        "--strict",
        action="store_true",
        help="Validate the generated file with provenance and artifact-consistency requirements.",
    )
    tcga_export = sub.add_parser(
        "tcga-export-artifacts",
        help="Convert a flat role-tagged training result CSV into TCGA artifact CSVs.",
    )
    tcga_export.add_argument("--input-csv", required=True, help="CSV with a role column: main, ablation, sensitivity, stats.")
    tcga_export.add_argument("--output-dir", default=r"D:\code\agent\example\results\logs")
    tcga_export.add_argument("--role-column", default="role", help="Column that identifies each row role.")
    tcga_export.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_export.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_export.add_argument("--metric", default="C-index")
    tcga_export.add_argument("--seed", default="2026")
    tcga_export.add_argument("--force", action="store_true", help="Overwrite existing export files.")
    tcga_export.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write available roles even if ablation/sensitivity/statistical rows are missing.",
    )
    tcga_demo_flow = sub.add_parser(
        "tcga-demo-artifact-flow",
        help="Run the bundled flat-CSV artifact demo through strict TCGA result generation.",
    )
    tcga_demo_flow.add_argument("--input-csv", default="examples/tcga_training_summary.csv")
    tcga_demo_flow.add_argument("--output-dir", default="outputs/tcga-artifact-flow")
    tcga_demo_flow.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_demo_flow.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_demo_flow.add_argument("--metric", default="C-index")
    tcga_demo_flow.add_argument("--seed", default="2026")
    tcga_demo_flow.add_argument("--force", action="store_true", help="Overwrite existing demo outputs.")
    tcga_demo_flow.add_argument(
        "--summary",
        default="",
        help="Optional JSON summary path. Defaults to output-dir/RUN_SUMMARY.json.",
    )
    tcga_demo_paper = sub.add_parser(
        "tcga-demo-paper",
        help="Run the bundled TCGA artifact-flow demo and draft a complete local paper package.",
    )
    tcga_demo_paper.add_argument("--example-root", default=r"D:\code\agent\example")
    tcga_demo_paper.add_argument("--input-csv", default="examples/tcga_training_summary.csv")
    tcga_demo_paper.add_argument("--output-dir", default="outputs/tcga-demo-paper")
    tcga_demo_paper.add_argument("--zip", default="", help="Optional Overleaf zip path. Defaults under output-dir.")
    tcga_demo_paper.add_argument("--target-venue", default="TPAMI")
    tcga_demo_paper.add_argument("--project-name", default="")
    tcga_demo_paper.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_demo_paper.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_demo_paper.add_argument("--metric", default="C-index")
    tcga_demo_paper.add_argument("--seed", default="2026")
    tcga_demo_paper.add_argument("--force", action="store_true", help="Overwrite existing demo artifact outputs.")
    tcga_demo_paper.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the configured LLM for section drafting. Default is deterministic local drafting.",
    )
    tcga_demo_paper.add_argument(
        "--min-llm-sections",
        type=int,
        default=4,
        help="Minimum LLM-written sections when --use-llm is enabled.",
    )
    tcga_demo_paper.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass when --use-llm is enabled.",
    )
    tcga_demo_paper.add_argument("--compile-latex", action="store_true", help="Run local LaTeX compile validation.")
    tcga_demo_paper.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    tcga_demo_paper.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    tcga_demo_paper_network = tcga_demo_paper.add_mutually_exclusive_group()
    tcga_demo_paper_network.add_argument("--online", action="store_true", help="Allow template/reference network calls.")
    tcga_demo_paper_network.add_argument("--offline", action="store_true", help="Disable template/reference network calls.")
    tcga_demo_paper.add_argument("--keyword", action="append", default=[], help="Additional keyword; can be repeated.")
    _add_experiment_contract_options(tcga_demo_paper)
    _add_experiment_quality_options(tcga_demo_paper)
    _add_experiment_provenance_options(tcga_demo_paper)
    _add_experiment_artifact_consistency_options(tcga_demo_paper)
    tcga_artifacts_doctor = sub.add_parser(
        "tcga-artifacts-doctor",
        help="Diagnose TCGA result CSV artifacts before generating the result Markdown file.",
    )
    tcga_artifacts_doctor.add_argument("--example-root", default=r"D:\code\agent\example")
    tcga_artifacts_doctor.add_argument(
        "--artifacts-dir",
        default="",
        help="Directory to scan recursively for TCGA result CSV artifacts.",
    )
    tcga_artifacts_doctor.add_argument("--main-csv", default="", help="Main result CSV; wide or long format.")
    tcga_artifacts_doctor.add_argument("--ablation-csv", default="", help="Optional ablation CSV.")
    tcga_artifacts_doctor.add_argument("--sensitivity-csv", default="", help="Optional sensitivity CSV.")
    tcga_artifacts_doctor.add_argument("--stats-csv", default="", help="Optional statistical-test CSV.")
    tcga_artifacts_doctor.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_artifacts_doctor.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_artifacts_doctor.add_argument("--metric", default="C-index")
    tcga_artifacts_doctor.add_argument("--dataset", action="append", default=[], help="Dataset/cohort name; can be repeated.")
    tcga_artifacts_doctor.add_argument("--summary", default="", help="Optional JSON artifact doctor summary path.")
    _add_experiment_contract_options(tcga_artifacts_doctor)
    tcga_artifact_template = sub.add_parser(
        "tcga-artifact-template",
        help="Write TCGA result CSV export templates and an artifact export contract.",
    )
    tcga_artifact_template.add_argument("--output-dir", default=r"D:\code\agent\example\results\logs")
    tcga_artifact_template.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_artifact_template.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_artifact_template.add_argument("--metric", default="C-index")
    tcga_artifact_template.add_argument("--seed", default="2026")
    tcga_artifact_template.add_argument("--dataset", action="append", default=[], help="Dataset/cohort name; can be repeated.")
    tcga_artifact_template.add_argument(
        "--style",
        choices=("long", "wide"),
        default="long",
        help="CSV schema style to generate. Long keeps fold/seed columns; wide is compact for paper tables.",
    )
    tcga_artifact_template.add_argument("--force", action="store_true", help="Overwrite existing template files.")
    tcga_preflight = sub.add_parser(
        "tcga-preflight",
        help="Run one read-only preflight report for TCGA paper generation readiness.",
    )
    tcga_preflight.add_argument("--example-root", default=r"D:\code\agent\example")
    tcga_preflight.add_argument(
        "--experiment-results",
        default="",
        help="TCGA result file. Defaults to example-root/results/tcga_results.md.",
    )
    tcga_preflight.add_argument(
        "--artifacts-dir",
        default="",
        help="Directory to scan recursively for TCGA result CSV artifacts.",
    )
    tcga_preflight.add_argument("--main-csv", default="", help="Main result CSV; wide or long format.")
    tcga_preflight.add_argument("--ablation-csv", default="", help="Optional ablation CSV.")
    tcga_preflight.add_argument("--sensitivity-csv", default="", help="Optional sensitivity CSV.")
    tcga_preflight.add_argument("--stats-csv", default="", help="Optional statistical-test CSV.")
    tcga_preflight.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_preflight.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_preflight.add_argument("--metric", default="C-index")
    tcga_preflight.add_argument("--dataset", action="append", default=[], help="Dataset/cohort name; can be repeated.")
    tcga_preflight.add_argument("--summary", default="", help="Optional JSON preflight summary path.")
    tcga_preflight.add_argument("--live-llm", action="store_true", help="Call the configured LLM during preflight.")
    tcga_preflight.add_argument("--disable-llm", action="store_true", help="Plan a deterministic local run without LLM calls.")
    tcga_preflight.add_argument("--compile-latex", action="store_true", help="Require a local LaTeX compiler.")
    tcga_preflight.add_argument("--submission-grade", action="store_true", help="Require submission-grade readiness.")
    tcga_preflight_network = tcga_preflight.add_mutually_exclusive_group()
    tcga_preflight_network.add_argument("--online", action="store_true", help="Plan online template/reference calls.")
    tcga_preflight_network.add_argument("--offline", action="store_true", help="Plan offline template/reference calls.")
    _add_experiment_contract_options(tcga_preflight)
    _add_experiment_quality_options(tcga_preflight)
    _add_experiment_provenance_options(tcga_preflight)
    _add_experiment_artifact_consistency_options(tcga_preflight)
    tcga_draft = sub.add_parser(
        "tcga-draft",
        help="Run the local Hyper-ProtoSurv TCGA paper path with a real result file.",
    )
    tcga_draft.add_argument("--example-root", default=r"D:\code\agent\example")
    tcga_draft.add_argument(
        "--experiment-results",
        default="",
        help="Real TCGA result file. Defaults to example-root/results/tcga_results.md.",
    )
    tcga_draft.add_argument(
        "--artifact-flow-summary",
        default="",
        help=(
            "Optional RUN_SUMMARY.json from tcga-demo-artifact-flow. "
            "When --experiment-results is omitted, tcga-draft uses the summary's experiment_results path."
        ),
    )
    tcga_draft.add_argument(
        "--project-name",
        default="",
        help="Project name for generated LaTeX artifacts. Defaults to the output directory name.",
    )
    tcga_draft.add_argument("--target-venue", default="TPAMI")
    tcga_draft.add_argument("--output-dir", default="outputs/hyper-protosurv-tcga-real")
    tcga_draft.add_argument("--zip", default="outputs/hyper-protosurv-tcga-real-overleaf.zip")
    tcga_draft.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    tcga_draft.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    tcga_network = tcga_draft.add_mutually_exclusive_group()
    tcga_network.add_argument(
        "--online",
        action="store_true",
        help="Allow template/reference network calls. The default TCGA draft run is offline.",
    )
    tcga_network.add_argument(
        "--offline",
        action="store_true",
        help="Disable template/reference network calls.",
    )
    tcga_draft.add_argument(
        "--disable-llm",
        action="store_true",
        help="Run deterministic local section drafting instead of requiring the configured LLM.",
    )
    tcga_draft.add_argument(
        "--min-llm-sections",
        type=int,
        default=4,
        help="Minimum number of paper sections that must be generated by the LLM.",
    )
    tcga_draft.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass for this run.",
    )
    tcga_draft.add_argument(
        "--compile-latex",
        action="store_true",
        help="Run the local LaTeX compiler during submission validation.",
    )
    tcga_draft.add_argument(
        "--submission-grade",
        action="store_true",
        help=(
            "Run the strict TCGA acceptance path: online references/templates, required LLM drafting, "
            "LLM self-review, LaTeX compilation, provenance checks, and artifact consistency checks."
        ),
    )
    tcga_draft.add_argument("--keyword", action="append", default=[], help="Additional keyword; can be repeated.")
    _add_experiment_contract_options(tcga_draft)
    _add_experiment_quality_options(tcga_draft)
    _add_experiment_provenance_options(tcga_draft)
    _add_experiment_artifact_consistency_options(tcga_draft)
    tcga_pipeline = sub.add_parser(
        "tcga-pipeline",
        help="Generate TCGA result evidence from artifacts, run doctor checks, then draft the paper.",
    )
    tcga_pipeline.add_argument("--example-root", default=r"D:\code\agent\example")
    tcga_pipeline.add_argument(
        "--experiment-results",
        default="",
        help="Generated/used TCGA result file. Defaults to example-root/results/tcga_results.md.",
    )
    tcga_pipeline.add_argument(
        "--artifacts-dir",
        default="",
        help="Directory to scan recursively for TCGA result CSV artifacts.",
    )
    tcga_pipeline.add_argument("--main-csv", default="", help="Main result CSV; wide or long format.")
    tcga_pipeline.add_argument("--ablation-csv", default="", help="Optional ablation CSV.")
    tcga_pipeline.add_argument("--sensitivity-csv", default="", help="Optional sensitivity CSV.")
    tcga_pipeline.add_argument("--stats-csv", default="", help="Optional statistical-test CSV.")
    tcga_pipeline.add_argument("--method", default="Hyper-ProtoSurv ours")
    tcga_pipeline.add_argument("--baseline", default="ProtoSurv baseline")
    tcga_pipeline.add_argument("--metric", default="C-index")
    tcga_pipeline.add_argument("--dataset", action="append", default=[], help="Dataset/cohort name; can be repeated.")
    tcga_pipeline.add_argument(
        "--skip-result-generation",
        action="store_true",
        help="Use the existing experiment-results file instead of generating it from CSV artifacts.",
    )
    tcga_pipeline.add_argument(
        "--write-artifact-template",
        action="store_true",
        help="If result CSV artifacts are missing, write TCGA artifact templates and stop before drafting.",
    )
    tcga_pipeline.add_argument(
        "--artifact-template-style",
        choices=("long", "wide"),
        default="long",
        help="CSV template style used with --write-artifact-template.",
    )
    tcga_pipeline.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Skip TCGA doctor checks before drafting.",
    )
    tcga_pipeline.add_argument(
        "--live-llm-doctor",
        action="store_true",
        help="Call the configured LLM during the doctor stage before drafting.",
    )
    tcga_pipeline.add_argument(
        "--project-name",
        default="",
        help="Project name for generated LaTeX artifacts. Defaults to the output directory name.",
    )
    tcga_pipeline.add_argument("--target-venue", default="TPAMI")
    tcga_pipeline.add_argument("--output-dir", default="outputs/hyper-protosurv-tcga-pipeline")
    tcga_pipeline.add_argument("--zip", default="outputs/hyper-protosurv-tcga-pipeline-overleaf.zip")
    tcga_pipeline.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    tcga_pipeline.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    tcga_pipeline_network = tcga_pipeline.add_mutually_exclusive_group()
    tcga_pipeline_network.add_argument(
        "--online",
        action="store_true",
        help="Allow template/reference network calls.",
    )
    tcga_pipeline_network.add_argument(
        "--offline",
        action="store_true",
        help="Disable template/reference network calls.",
    )
    tcga_pipeline.add_argument(
        "--disable-llm",
        action="store_true",
        help="Run deterministic local section drafting instead of requiring the configured LLM.",
    )
    tcga_pipeline.add_argument(
        "--min-llm-sections",
        type=int,
        default=4,
        help="Minimum number of paper sections that must be generated by the LLM.",
    )
    tcga_pipeline.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass for this run.",
    )
    tcga_pipeline.add_argument(
        "--compile-latex",
        action="store_true",
        help="Run the local LaTeX compiler during submission validation.",
    )
    tcga_pipeline.add_argument(
        "--submission-grade",
        action="store_true",
        help="Run the strict TCGA acceptance path after generating result evidence.",
    )
    tcga_pipeline.add_argument("--keyword", action="append", default=[], help="Additional keyword; can be repeated.")
    _add_experiment_contract_options(tcga_pipeline)
    _add_experiment_quality_options(tcga_pipeline)
    _add_experiment_provenance_options(tcga_pipeline)
    _add_experiment_artifact_consistency_options(tcga_pipeline)
    tcga_readiness_schema = sub.add_parser(
        "tcga-readiness-schema",
        help="Print or write the versioned JSON schema/example for TCGA readiness_contract payloads.",
    )
    tcga_readiness_schema.add_argument("--output", default="", help="Optional path to write JSON.")
    tcga_readiness_schema.add_argument(
        "--example",
        action="store_true",
        help="Write an example readiness_contract instead of the schema.",
    )
    sub.add_parser("llm-ping", help="Test the configured OpenAI-compatible LLM.")
    llm_doctor = sub.add_parser("llm-doctor", help="Inspect LLM configuration and provider health.")
    llm_doctor.add_argument(
        "--no-live",
        action="store_true",
        help="Only print local configuration; do not call the provider.",
    )
    llm_doctor.add_argument("--summary", default="", help="Optional JSON summary path for provider diagnostics.")
    llm_live = sub.add_parser(
        "llm-live-smoke",
        help="Run one explicit live LLM call and write a reproducible diagnostic summary.",
    )
    llm_live.add_argument("--summary", default="outputs/llm-live-smoke.json", help="JSON summary path.")
    llm_live.add_argument("--prompt", default="Reply with exactly: paper-agent-live-ok")
    llm_live.add_argument("--expect", default="paper-agent-live-ok")
    llm_live.add_argument("--max-tokens", type=int, default=32)
    llm_live.add_argument("--temperature", type=float, default=0.0)
    sub.add_parser("llm-self-review-smoke", help="Run a tiny configured-LLM self-review smoke test.")
    sub.add_parser("latex-doctor", help="Check local LaTeX compiler availability and install guidance.")
    llm_draft = sub.add_parser(
        "llm-draft-smoke",
        help="Run a full local draft smoke test and require configured LLM section calls.",
    )
    llm_draft.add_argument("--example-root", default=r"D:\code\agent\example")
    llm_draft.add_argument(
        "--project-name",
        default="",
        help="Project name for generated LaTeX artifacts. Defaults to the output directory name.",
    )
    llm_draft.add_argument("--target-venue", default="TPAMI")
    llm_draft.add_argument(
        "--experiment-results",
        default="examples/hyper_protosurv_mock_experiments.md",
        help="Experiment table file for the smoke run.",
    )
    llm_draft.add_argument("--output-dir", default="outputs/llm-draft-smoke")
    llm_draft.add_argument("--zip", default="outputs/llm-draft-smoke-overleaf.zip")
    llm_draft.add_argument(
        "--min-llm-sections",
        type=int,
        default=4,
        help="Minimum number of paper sections that must be generated by the LLM.",
    )
    llm_draft.add_argument(
        "--include-llm-self-review",
        action="store_true",
        help="Also require the final LLM self-review pass to complete.",
    )
    llm_draft.add_argument(
        "--online",
        action="store_true",
        help="Allow template/reference network calls. The default smoke run keeps those offline.",
    )
    llm_draft.add_argument(
        "--compile-latex",
        action="store_true",
        help="Run the local LaTeX compiler during submission validation.",
    )
    llm_draft.add_argument(
        "--strict-results",
        action="store_true",
        help="Fail before LLM generation unless the experiment input is real and contract-complete.",
    )
    _add_experiment_contract_options(llm_draft)
    _add_experiment_provenance_options(llm_draft)
    _add_experiment_artifact_consistency_options(llm_draft)
    paper_smoke = sub.add_parser(
        "paper-e2e-smoke",
        help="Run explicit code+baseline+venue+results input-to-paper smoke test.",
    )
    paper_smoke.add_argument("--project-name", default="")
    paper_smoke.add_argument(
        "--baseline-pdf",
        required=True,
        help="Baseline PDF path or directory containing a PDF.",
    )
    paper_smoke.add_argument("--code-path", required=True, help="Project code directory.")
    paper_smoke.add_argument(
        "--experiment-results",
        required=True,
        help="Markdown/CSV/text experiment results file.",
    )
    paper_smoke.add_argument("--target-venue", required=True, help="Target journal or conference.")
    paper_smoke.add_argument("--keyword", action="append", default=[], help="Keyword; can be repeated.")
    paper_smoke.add_argument("--output-dir", default="outputs/paper-e2e-smoke")
    paper_smoke.add_argument("--zip", default="outputs/paper-e2e-smoke-overleaf.zip")
    paper_smoke.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    paper_smoke.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    paper_smoke_network = paper_smoke.add_mutually_exclusive_group()
    paper_smoke_network.add_argument(
        "--online",
        action="store_true",
        help="Allow template/reference/related-work network calls.",
    )
    paper_smoke_network.add_argument(
        "--offline",
        action="store_true",
        help="Disable template/reference/related-work network calls.",
    )
    paper_smoke.add_argument(
        "--require-llm",
        action="store_true",
        help="Require configured live LLM preflight and section generation.",
    )
    paper_smoke.add_argument(
        "--min-llm-sections",
        type=int,
        default=0,
        help="Fail unless at least this many sections are written by the LLM.",
    )
    paper_smoke.add_argument(
        "--include-llm-self-review",
        action="store_true",
        help="Require the final LLM self-review pass. Usually pair with --require-llm.",
    )
    paper_smoke.add_argument(
        "--compile-latex",
        action="store_true",
        help="Run the local LaTeX compiler during submission validation.",
    )
    paper_smoke.add_argument(
        "--strict-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate the experiment result file strictly before generation.",
    )
    paper_smoke.add_argument(
        "--write-artifact-template",
        action="store_true",
        help="When strict result preflight blocks, write TCGA result CSV templates.",
    )
    paper_smoke.add_argument(
        "--artifact-template-dir",
        default="",
        help="Directory for generated result CSV templates. Defaults to experiment-results parent/logs.",
    )
    paper_smoke.add_argument(
        "--artifact-template-force",
        action="store_true",
        help="Overwrite existing generated artifact-template files.",
    )
    paper_smoke.add_argument(
        "--artifact-template-style",
        choices=("long", "wide"),
        default="long",
        help="CSV schema style for --write-artifact-template.",
    )
    paper_smoke.add_argument("--artifact-template-seed", default="2026")
    paper_smoke.add_argument(
        "--artifact-template-dataset",
        action="append",
        default=[],
        help="Dataset/cohort name for artifact templates; can be repeated.",
    )
    paper_smoke.add_argument("--artifact-method", default="Hyper-ProtoSurv ours")
    paper_smoke.add_argument("--artifact-baseline", default="ProtoSurv baseline")
    paper_smoke.add_argument("--artifact-metric", default="C-index")
    paper_smoke.add_argument(
        "--generate-results-from-artifacts",
        action="store_true",
        help="Generate experiment-results Markdown from completed TCGA result CSV artifacts before validation.",
    )
    paper_smoke.add_argument(
        "--artifacts-dir",
        default="",
        help="Completed result CSV artifact directory for --generate-results-from-artifacts.",
    )
    _add_experiment_contract_options(paper_smoke)
    _add_experiment_provenance_options(paper_smoke)
    _add_experiment_artifact_consistency_options(paper_smoke)
    experiment_template = sub.add_parser(
        "experiment-template",
        help="Write a fill-in Markdown template for real experiment result files.",
    )
    experiment_template.add_argument("--output", default="", help="Optional path to write the template.")
    experiment_template.add_argument("--method", default="Hyper-ProtoSurv ours")
    experiment_template.add_argument("--baseline", default="ProtoSurv baseline")
    experiment_template.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset/cohort name; can be repeated. Defaults to common TCGA cohorts.",
    )
    validate_results = sub.add_parser(
        "validate-results",
        help="Validate an experiment result file without generating a paper.",
    )
    validate_results.add_argument("--experiment-results", required=True)
    validate_results.add_argument("--summary", default="", help="Optional JSON summary path.")
    validate_results.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status unless the result source is real and the contract is complete.",
    )
    _add_experiment_contract_options(validate_results)
    _add_experiment_quality_options(validate_results)
    _add_experiment_provenance_options(validate_results)
    _add_experiment_artifact_consistency_options(validate_results)
    args = parser.parse_args()

    if args.command == "validate-results":
        summary = _validate_results_file(
            Path(args.experiment_results),
            summary_path=Path(args.summary) if args.summary else None,
            **_experiment_contract_kwargs(args),
            **_experiment_quality_kwargs(args),
            **_experiment_provenance_kwargs(args),
            **_experiment_artifact_consistency_kwargs(args),
        )
        if args.strict and not _validated_results_are_strictly_acceptable(summary):
            raise SystemExit("Experiment result validation failed in strict mode.")
    elif args.command == "experiment-template":
        template = experiment_results_template(
            method=args.method,
            baseline=args.baseline,
            datasets=args.dataset or None,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(template, encoding="utf-8")
            print(f"Experiment result template written to {output_path}")
        else:
            print(template)
    elif args.command == "demo":
        request = PaperRequest(
            project_name="adaptive-baseline-improvement",
            target_venue="IEEE Conference",
            method_notes=(
                "Adaptive representation calibration module\n"
                "Uncertainty-aware training objective\n"
                "Lightweight inference path for deployment"
            ),
            experiment_results=(
                "| Method | DatasetA Accuracy | DatasetB F1 |\n"
                "|---|---:|---:|\n"
                "| baseline | 81.2 | 74.5 |\n"
                "| ours | 84.6 | 77.1 |\n"
                "Ablation w/o calibration drops performance."
            ),
            keywords=["representation", "uncertainty", "efficient inference"],
            template_zip_path=args.template_zip or None,
            template_dir_path=args.template_dir or None,
            skip_llm_self_review=args.skip_llm_self_review,
        )
        state = PaperWorkflow().run(request)
        state.setdefault("artifacts", {})["experiment_results_source"] = "inline_demo"
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        markdown_path = output / "draft.md"
        markdown_path.write_text(state["final_markdown"], encoding="utf-8")
        print(f"Draft written to {markdown_path}")
        print(f"Template source: {state['venue_template'].template_source}")
        print(f"Bibliography entries: {len(state.get('bibliography', []))}")
        print(f"LaTeX tables: {state.get('artifacts', {}).get('latex_table_count', 0)}")
        print(f"LLM self-review: {_llm_self_review_mode(state)}")
        print(f"LaTeX written to {state['latex_output_path']}")
        if args.zip:
            zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
            print(f"Overleaf zip written to {zip_path}")
        summary_path, acceptance_report_path = _write_run_reports(
            state,
            summary_path=Path(args.summary) if args.summary else None,
            markdown_path=markdown_path,
            acceptance_report_path=Path(args.acceptance_report) if args.acceptance_report else None,
            default_acceptance_report=False,
            min_llm_sections=0,
        )
        if summary_path:
            print(f"Run summary written to {summary_path}")
        if acceptance_report_path:
            print(f"Acceptance report written to {acceptance_report_path}")
    elif args.command == "draft":
        network_mode = _configure_network_mode(args)
        llm_mode = _configure_llm_mode(args)
        compile_latex_requested = _configure_latex_compile(args)
        runtime_llm_config = load_llm_config()
        baseline_pdf = _resolve_baseline_pdf(args.baseline)
        experiment_path = _resolve_project_relative_path(args.experiment_results)
        if not experiment_path.is_file():
            raise SystemExit(f"Experiment results file not found: {experiment_path}")
        experiment_results = experiment_path.read_text(encoding="utf-8")
        result_preflight = _validate_results_text(
            experiment_path,
            experiment_results,
            **_experiment_contract_kwargs(args),
            **_experiment_provenance_kwargs(args),
            **_experiment_artifact_consistency_kwargs(args),
        )
        if args.strict_results and not _validated_results_are_strictly_acceptable(result_preflight):
            raise SystemExit("Draft failed: experiment result validation failed in strict mode.")
        request = PaperRequest(
            project_name=args.project_name,
            target_venue=args.target_venue,
            baseline_pdf_path=str(baseline_pdf),
            code_path=args.code_path,
            template_zip_path=args.template_zip or None,
            template_dir_path=args.template_dir or None,
            experiment_results=experiment_results,
            keywords=args.keyword,
            skip_llm_self_review=args.skip_llm_self_review,
        )
        state = PaperWorkflow().run(request)
        _record_runtime_modes(
            state,
            network_mode=network_mode,
            llm_mode=llm_mode,
            compile_latex_requested=compile_latex_requested,
            min_llm_sections=args.min_llm_sections,
            llm_config=runtime_llm_config,
        )
        state.setdefault("artifacts", {})["experiment_results_source"] = "file"
        state["artifacts"]["experiment_results_path"] = str(experiment_path)
        _record_result_preflight(state, result_preflight)
        SubmissionReadinessAgent().run(state)
        DraftReportAgent().run(state)
        markdown_path = None
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(state["final_markdown"], encoding="utf-8")
            markdown_path = output_path
            print(f"Markdown written to {output_path}")
        print(f"Section writer mode: {state.get('artifacts', {}).get('section_writer_mode')}")
        section_errors = state.get("artifacts", {}).get("section_writer_section_errors", {})
        if section_errors:
            print(f"Section errors: {section_errors}")
        guard_findings = state.get("artifacts", {}).get("evidence_guard_findings", [])
        print(f"Evidence guard findings: {len(guard_findings)}")
        print(f"Review findings: {len(state.get('review_findings', []))}")
        print(f"Template source: {state['venue_template'].template_source}")
        print(f"Bibliography entries: {len(state.get('bibliography', []))}")
        print(f"LaTeX tables: {state.get('artifacts', {}).get('latex_table_count', 0)}")
        print(f"Network mode: {network_mode}")
        print(f"LLM mode: {llm_mode}")
        print(f"LLM self-review: {_llm_self_review_mode(state)}")
        print(f"LaTeX written to {state['latex_output_path']}")
        if args.zip:
            zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
            print(f"Overleaf zip written to {zip_path}")
        summary_path, acceptance_report_path = _write_run_reports(
            state,
            summary_path=Path(args.summary) if args.summary else None,
            markdown_path=markdown_path,
            acceptance_report_path=Path(args.acceptance_report) if args.acceptance_report else None,
            min_llm_sections=args.min_llm_sections,
        )
        if summary_path:
            print(f"Run summary written to {summary_path}")
        if acceptance_report_path:
            print(f"Acceptance report written to {acceptance_report_path}")
        successes = state.get("artifacts", {}).get("section_writer_llm_successes", [])
        if len(successes) < args.min_llm_sections:
            raise SystemExit(
                f"Draft failed: expected at least {args.min_llm_sections} LLM-written sections, "
                f"got {len(successes)}."
            )
    elif args.command == "sample-hyper-protosurv":
        _run_hyper_protosurv_sample(args)
    elif args.command == "tcga-doctor":
        _run_tcga_doctor(args)
    elif args.command == "tcga-results-from-artifacts":
        _run_tcga_results_from_artifacts(args)
    elif args.command == "tcga-export-artifacts":
        _run_tcga_export_artifacts(args)
    elif args.command == "tcga-demo-artifact-flow":
        _run_tcga_demo_artifact_flow(args)
    elif args.command == "tcga-demo-paper":
        _run_tcga_demo_paper(args)
    elif args.command == "tcga-artifacts-doctor":
        _run_tcga_artifacts_doctor(args)
    elif args.command == "tcga-artifact-template":
        _run_tcga_artifact_template(args)
    elif args.command == "tcga-preflight":
        _run_tcga_preflight(args)
    elif args.command == "tcga-draft":
        _run_tcga_draft(args)
    elif args.command == "tcga-pipeline":
        _run_tcga_pipeline(args)
    elif args.command == "tcga-readiness-schema":
        _run_tcga_readiness_schema(args)
    elif args.command == "llm-ping":
        config = load_llm_config()
        client = LLMClient(config)
        try:
            result = client.chat(
                [
                    ChatMessage(role="system", content="You are a concise API health-check assistant."),
                    ChatMessage(role="user", content="Reply with exactly: paper-agent-ok"),
                ],
                temperature=0,
                max_tokens=128,
            )
        except LLMError as exc:
            raise SystemExit(f"LLM ping failed: {exc}") from exc
        print(result.content.strip())
    elif args.command == "llm-doctor":
        _run_llm_doctor(args)
    elif args.command == "llm-live-smoke":
        _run_llm_live_smoke(args)
    elif args.command == "llm-self-review-smoke":
        _run_llm_self_review_smoke()
    elif args.command == "latex-doctor":
        _run_latex_doctor()
    elif args.command == "llm-draft-smoke":
        _run_llm_draft_smoke(args)
    elif args.command == "paper-e2e-smoke":
        _run_paper_e2e_smoke(args)


def _resolve_baseline_pdf(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
    raise SystemExit(f"No baseline PDF found at {path}")


def _llm_self_review_mode(state: dict) -> str:
    return str(state.get("artifacts", {}).get("llm_self_review", {}).get("mode", "not run"))


NETWORK_DISABLE_ENV_VARS = (
    "PAPER_AGENT_DISABLE_TEMPLATE_FETCH",
    "PAPER_AGENT_DISABLE_REFERENCE_RESOLVE",
    "PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY",
)
LLM_API_KEY_ENV_VARS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ARK_API_KEY")
LLM_BASE_URL_ENV_VARS = ("DEEPSEEK_API_BASE", "OPENAI_API_BASE")


def _configure_network_mode(args: argparse.Namespace, *, default_offline: bool = False) -> str:
    if getattr(args, "online", False):
        for name in NETWORK_DISABLE_ENV_VARS:
            os.environ[name] = "0"
        return "online"
    if getattr(args, "offline", False) or default_offline:
        for name in NETWORK_DISABLE_ENV_VARS:
            os.environ[name] = "1"
        return "offline"
    return "environment"


def _configure_llm_mode(args: argparse.Namespace, *, default_disabled: bool = False) -> str:
    if getattr(args, "allow_llm", False):
        os.environ["PAPER_AGENT_DISABLE_LLM"] = "0"
        return "enabled"
    if getattr(args, "disable_llm", False) or default_disabled:
        os.environ["PAPER_AGENT_DISABLE_LLM"] = "1"
        return "disabled"
    return "environment"


def _configure_latex_compile(args: argparse.Namespace) -> bool:
    if getattr(args, "compile_latex", False):
        os.environ["PAPER_AGENT_RUN_LATEX_COMPILE"] = "1"
        return True
    return _truthy_env("PAPER_AGENT_RUN_LATEX_COMPILE")


def _apply_submission_grade_defaults(args: argparse.Namespace) -> bool:
    if not getattr(args, "submission_grade", False):
        return False
    if getattr(args, "offline", False):
        raise SystemExit("TCGA submission-grade runs require online mode; remove --offline.")
    if getattr(args, "disable_llm", False):
        raise SystemExit("TCGA submission-grade runs require the configured LLM; remove --disable-llm.")
    if getattr(args, "skip_llm_self_review", False):
        raise SystemExit(
            "TCGA submission-grade runs require LLM self-review; remove --skip-llm-self-review."
        )
    args.online = True
    args.compile_latex = True
    args.require_provenance = True
    args.require_artifact_consistency = True
    args.min_llm_sections = max(int(getattr(args, "min_llm_sections", 0) or 0), 4)
    return True


def _record_runtime_modes(
    state: dict,
    *,
    network_mode: str,
    llm_mode: str,
    compile_latex_requested: bool,
    min_llm_sections: int = 0,
    llm_config: LLMConfig | None = None,
    submission_grade: bool = False,
) -> None:
    artifacts = state.setdefault("artifacts", {})
    artifacts["runtime_network_mode"] = network_mode
    artifacts["runtime_llm_mode"] = llm_mode
    artifacts["latex_compile_requested"] = compile_latex_requested
    artifacts["min_llm_sections"] = min_llm_sections
    artifacts["submission_grade"] = submission_grade
    if llm_config:
        artifacts.update(_llm_runtime_metadata(llm_config))


def _llm_runtime_metadata(config: LLMConfig) -> dict[str, object]:
    parsed = urlparse(config.base_url)
    host = parsed.netloc or parsed.path.split("/")[0]
    return {
        "runtime_llm_provider": _llm_provider_from_host(host),
        "runtime_llm_model": config.model,
        "runtime_llm_endpoint_host": host,
        "runtime_llm_configured": config.configured,
        "runtime_llm_timeout_seconds": config.timeout_seconds,
        "runtime_llm_max_retries": config.max_retries,
    }


def _llm_provider_from_host(host: str) -> str:
    lowered = host.lower()
    if "deepseek" in lowered:
        return "deepseek"
    if "volcengine" in lowered or "ark" in lowered:
        return "volcengine-ark"
    if "dashscope" in lowered or "aliyuncs" in lowered or "bailian" in lowered:
        return "aliyun-bailian"
    if "openai" in lowered:
        return "openai"
    return host or "unknown"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _llm_preflight_check(client: LLMClient, config: LLMConfig, *, context: str) -> dict[str, object]:
    started = time.perf_counter()
    try:
        result = client.chat(
            [
                ChatMessage(role="system", content="You are a concise API health-check assistant."),
                ChatMessage(role="user", content="Reply with exactly: paper-agent-ok"),
            ],
            temperature=0,
            max_tokens=16,
        )
    except LLMError as exc:
        raise SystemExit(_llm_preflight_failure_message(config, context, exc)) from exc
    if result.content.strip() != "paper-agent-ok":
        raise SystemExit(
            f"{context} LLM preflight failed: unexpected provider response "
            f"from {_llm_config_label(config)}: {result.content.strip()[:120]!r}."
        )
    elapsed_seconds = max(0.0, time.perf_counter() - started)
    return _llm_preflight_success_diagnostics(result, elapsed_seconds)


def _llm_preflight_success_diagnostics(result: object, elapsed_seconds: float) -> dict[str, object]:
    usage = getattr(result, "usage", {})
    return {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "response_model": str(getattr(result, "model", "")),
        "usage": usage if isinstance(usage, dict) else {},
    }


def _llm_preflight_failure_message(config: LLMConfig, context: str, exc: LLMError) -> str:
    raw = _sanitize_llm_error(str(exc), config)
    diagnosis = _llm_failure_diagnosis(raw)
    kind = _llm_failure_kind(raw)
    return (
        f"{context} LLM preflight failed for {_llm_config_label(config)}: "
        f"{diagnosis} Failure kind: {kind}. Raw provider error: {raw}"
    )


def _llm_config_label(config: LLMConfig) -> str:
    parsed = urlparse(config.base_url)
    host = parsed.netloc or parsed.path.split("/")[0]
    return f"{_llm_provider_from_host(host)}/{config.model} at {host or 'unknown-host'}"


def _llm_failure_diagnosis(raw_error: str) -> str:
    kind = _llm_failure_kind(raw_error)
    if kind == "configuration":
        return "LLM is not configured. Set an API key and TEXT_MODEL before running generation."
    if kind == "quota":
        return (
            "provider account balance or quota is insufficient. Recharge/enable billing, "
            "switch to another API key, or change provider/model before running generation."
        )
    if kind == "authentication":
        return "API authentication failed. Check the configured API key and provider base URL."
    if kind == "permission":
        return "the API key is not allowed to use this model or endpoint."
    if kind == "model_not_found":
        return "the configured model or endpoint was not found. Check TEXT_MODEL and API base URL."
    if kind == "timeout":
        return "the provider did not respond before the configured timeout."
    if kind == "transport":
        return "network transport failed before the provider returned a completion."
    return "the provider rejected or failed the health-check request."


def _llm_failure_kind(raw_error: str) -> str:
    lowered = raw_error.lower()
    if "requires a configured llm" in lowered or "not configured" in lowered or "no configured api key" in lowered:
        return "configuration"
    if (
        "http 402" in lowered
        or "insufficient balance" in lowered
        or "insufficient_balance" in lowered
        or "quota" in lowered
        or "billing" in lowered
    ):
        return "quota"
    if "http 401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return "authentication"
    if "http 403" in lowered or "forbidden" in lowered or "permission" in lowered:
        return "permission"
    if "http 404" in lowered or "model_not_found" in lowered or "not found" in lowered:
        return "model_not_found"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "transport error" in lowered or "connection" in lowered:
        return "transport"
    return "unknown"


def _llm_failure_diagnostics(config: LLMConfig, raw_error: str) -> dict[str, object]:
    sanitized = _sanitize_llm_error(raw_error, config)
    metadata = _llm_runtime_metadata(config)
    return {
        "failure_kind": _llm_failure_kind(sanitized),
        "diagnosis": _llm_failure_diagnosis(sanitized),
        "raw_error": sanitized,
        "provider": metadata["runtime_llm_provider"],
        "model": metadata["runtime_llm_model"],
        "endpoint_host": metadata["runtime_llm_endpoint_host"],
        "configured": metadata["runtime_llm_configured"],
        "timeout_seconds": metadata["runtime_llm_timeout_seconds"],
        "connect_timeout_seconds": config.connect_timeout_seconds,
        "max_retries": metadata["runtime_llm_max_retries"],
        "retry_base_seconds": config.retry_base_seconds,
    }


def _sanitize_llm_error(raw_error: str, config: LLMConfig) -> str:
    sanitized = raw_error
    candidates = [config.api_key, *[os.getenv(name, "").strip() for name in LLM_API_KEY_ENV_VARS]]
    for value in candidates:
        if value and len(value) >= 4:
            sanitized = sanitized.replace(value, "[redacted-api-key]")
    return sanitized


def _run_llm_doctor(args: argparse.Namespace) -> None:
    config = load_llm_config()
    print("LLM configuration:")
    print(f"- Provider/model: {_llm_config_label(config)}")
    print(f"- API key: {_env_source_label(LLM_API_KEY_ENV_VARS)}")
    print(f"- Base URL source: {_env_source_label(LLM_BASE_URL_ENV_VARS, default_label='default')}")
    print(f"- Model source: {_env_source_label(('TEXT_MODEL',), default_label='default')}")
    print(f"- Disabled by PAPER_AGENT_DISABLE_LLM: {_truthy_env('PAPER_AGENT_DISABLE_LLM')}")
    print(f"- Configured: {config.configured}")
    print(f"- Timeout: {config.timeout_seconds}s; connect={config.connect_timeout_seconds}s")
    print(f"- Max retries: {config.max_retries}; retry base={config.retry_base_seconds}s")
    print(f"- Max tokens: {config.max_tokens}; temperature={config.temperature}")
    print(f"- Thinking: {config.thinking}; reasoning_effort={config.reasoning_effort or 'not set'}")

    if args.no_live:
        print("Live preflight: SKIPPED (--no-live)")
        _write_llm_doctor_summary_if_requested(
            args,
            config,
            status="pass" if config.configured else "warn",
            live_status="skipped",
        )
        return
    if not config.configured:
        print("Live preflight: FAIL")
        _write_llm_doctor_summary_if_requested(
            args,
            config,
            status="fail",
            live_status="fail",
            live_error="LLM doctor failed: no configured API key/model.",
        )
        raise SystemExit(
            "LLM doctor failed: no configured API key/model. Set DEEPSEEK_API_KEY, "
            "OPENAI_API_KEY, or ARK_API_KEY plus TEXT_MODEL, and ensure PAPER_AGENT_DISABLE_LLM is not enabled."
        )

    client = LLMClient(config)
    try:
        live_result = _llm_preflight_check(client, config, context="LLM doctor")
    except SystemExit as exc:
        print("Live preflight: FAIL")
        _write_llm_doctor_summary_if_requested(
            args,
            config,
            status="fail",
            live_status="fail",
            live_error=str(exc),
        )
        raise exc
    print("Live preflight: PASS")
    _write_llm_doctor_summary_if_requested(
        args,
        config,
        status="pass",
        live_status="pass",
        live_result=live_result,
    )


def _write_llm_doctor_summary_if_requested(
    args: argparse.Namespace,
    config: LLMConfig,
    *,
    status: str,
    live_status: str,
    live_error: str = "",
    live_result: dict[str, object] | None = None,
) -> Path | None:
    summary_path = getattr(args, "summary", "")
    if not summary_path:
        return None
    summary = _llm_doctor_summary(
        config,
        status=status,
        live_status=live_status,
        live_error=live_error,
        live_result=live_result,
    )
    written = _write_run_summary_data(summary, Path(summary_path))
    print(f"LLM doctor summary written to {written}")
    return written


def _llm_doctor_summary(
    config: LLMConfig,
    *,
    status: str,
    live_status: str,
    live_error: str = "",
    live_result: dict[str, object] | None = None,
) -> dict[str, object]:
    live_preflight: dict[str, object] = {"status": live_status}
    if live_result:
        live_preflight.update(live_result)
    if live_error:
        live_preflight["diagnostics"] = _llm_failure_diagnostics(config, live_error)
    return {
        "status": status,
        "llm": _llm_static_summary(config),
        "live_preflight": live_preflight,
    }


def _run_llm_live_smoke(args: argparse.Namespace) -> None:
    config = load_llm_config()
    summary_path = Path(args.summary)
    request = _llm_live_smoke_request(args)
    print("LLM live smoke:")
    print(f"- Provider/model: {_llm_config_label(config)}")
    print(f"- Summary: {summary_path}")
    if not config.configured:
        diagnostics = _llm_failure_diagnostics(config, "LLM live smoke failed: no configured API key/model.")
        summary = _llm_live_smoke_summary(
            config,
            status="fail",
            request=request,
            live_call={"status": "fail", "diagnostics": diagnostics},
        )
        written = _write_run_summary_data(summary, summary_path)
        print(f"LLM live smoke summary written to {written}")
        raise SystemExit(
            "LLM live smoke failed: no configured API key/model. "
            "Run `paper-agent llm-doctor --no-live` to inspect configuration."
        )

    client = LLMClient(config)
    started = time.perf_counter()
    try:
        result = client.chat(
            [
                ChatMessage(role="system", content="You are a concise API smoke-test assistant."),
                ChatMessage(role="user", content=args.prompt),
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except LLMError as exc:
        elapsed = max(0.0, time.perf_counter() - started)
        diagnostics = _llm_failure_diagnostics(config, str(exc))
        live_call = {
            "status": "fail",
            "elapsed_seconds": round(elapsed, 3),
            "diagnostics": diagnostics,
        }
        summary = _llm_live_smoke_summary(config, status="fail", request=request, live_call=live_call)
        written = _write_run_summary_data(summary, summary_path)
        print("LLM live smoke: FAIL")
        print(f"LLM live smoke summary written to {written}")
        raise SystemExit(f"LLM live smoke failed: {diagnostics['diagnosis']} Summary: {written}") from exc

    elapsed = max(0.0, time.perf_counter() - started)
    response = result.content.strip()
    matched = response == args.expect
    live_call = {
        "status": "pass" if matched else "fail",
        "matched_expectation": matched,
        "elapsed_seconds": round(elapsed, 3),
        "response_model": str(result.model),
        "usage": result.usage if isinstance(result.usage, dict) else {},
        "response_preview": response[:240],
    }
    if not matched:
        live_call["diagnostics"] = {
            "failure_kind": "unexpected_response",
            "diagnosis": "The provider responded, but the content did not match the expected smoke-test string.",
        }
    status = "pass" if matched else "fail"
    summary = _llm_live_smoke_summary(config, status=status, request=request, live_call=live_call)
    written = _write_run_summary_data(summary, summary_path)
    print(f"LLM live smoke: {status.upper()}")
    print(f"Response preview: {response[:120]}")
    print(f"LLM live smoke summary written to {written}")
    if not matched:
        raise SystemExit(f"LLM live smoke failed: response did not match expected string. Summary: {written}")


def _llm_live_smoke_request(args: argparse.Namespace) -> dict[str, object]:
    prompt = str(args.prompt)
    return {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_preview": prompt[:120],
        "expected_response": str(args.expect),
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
    }


def _llm_live_smoke_summary(
    config: LLMConfig,
    *,
    status: str,
    request: dict[str, object],
    live_call: dict[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "llm": _llm_static_summary(config),
        "request": request,
        "live_call": live_call,
    }


def _llm_static_summary(config: LLMConfig) -> dict[str, object]:
    metadata = _llm_runtime_metadata(config)
    return {
        "provider": metadata["runtime_llm_provider"],
        "model": metadata["runtime_llm_model"],
        "endpoint_host": metadata["runtime_llm_endpoint_host"],
        "configured": metadata["runtime_llm_configured"],
        "api_key_source": _env_source_label(LLM_API_KEY_ENV_VARS),
        "base_url_source": _env_source_label(LLM_BASE_URL_ENV_VARS, default_label="default"),
        "model_source": _env_source_label(("TEXT_MODEL",), default_label="default"),
        "disabled_by_env": _truthy_env("PAPER_AGENT_DISABLE_LLM"),
        "timeout_seconds": metadata["runtime_llm_timeout_seconds"],
        "connect_timeout_seconds": config.connect_timeout_seconds,
        "max_retries": metadata["runtime_llm_max_retries"],
        "retry_base_seconds": config.retry_base_seconds,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "thinking": config.thinking,
        "reasoning_effort": config.reasoning_effort,
    }


def _env_source_label(names: tuple[str, ...], *, default_label: str = "not set") -> str:
    for name in names:
        if os.getenv(name, "").strip():
            return f"configured via {name}"
    return default_label


def _run_latex_doctor() -> None:
    status = _latex_toolchain_status()
    print("LaTeX toolchain:")
    for name, path in status["tools"].items():
        print(f"- {name}: {path or 'not found'}")
    if status["available"]:
        print(f"Preferred compiler: {status['preferred_tool']}")
        print("Compile checks can be enabled with --compile-latex.")
    else:
        print("No local LaTeX compiler found.")
        print(f"Install hint: {status['install_hint']}")


def _latex_toolchain_status() -> dict[str, object]:
    validator = SubmissionPackageValidatorAgent()
    tools = {
        "latexmk": validator._find_executable("latexmk"),
        "pdflatex": validator._find_executable("pdflatex"),
        "tectonic": validator._find_executable("tectonic"),
    }
    preferred = tools["latexmk"] or tools["pdflatex"] or tools["tectonic"]
    return {
        "tools": tools,
        "available": bool(preferred),
        "preferred_tool": Path(str(preferred)).name if preferred else "",
        "install_hint": validator._compile_install_hint(),
    }


def _refresh_submission_artifacts(state: dict) -> None:
    SubmissionPackageValidatorAgent().run(state)
    SubmissionReadinessAgent().run(state)
    DraftReportAgent().run(state)


def _write_latex_zip_and_refresh(state: dict, zip_path: Path) -> Path:
    DraftReportAgent().run(state)
    written_path = zip_latex_project(state["latex_project_dir"], zip_path)
    state["latex_zip_path"] = written_path
    _refresh_submission_artifacts(state)
    written_path = zip_latex_project(state["latex_project_dir"], zip_path)
    state["latex_zip_path"] = written_path
    SubmissionPackageValidatorAgent().run(state)
    SubmissionReadinessAgent().run(state)
    return written_path


def _run_hyper_protosurv_sample(args: argparse.Namespace) -> None:
    llm_mode = _configure_llm_mode(args, default_disabled=True)
    network_mode = _configure_network_mode(args, default_offline=True)
    compile_latex_requested = _configure_latex_compile(args)
    runtime_llm_config = load_llm_config()

    example_root = Path(args.example_root)
    baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
    code_path = example_root / "code" / "hyper-protosurv"
    if not code_path.is_dir():
        raise SystemExit(f"Hyper-ProtoSurv code directory not found: {code_path}")

    if args.experiment_results:
        experiment_path = _resolve_project_relative_path(args.experiment_results)
        if not experiment_path.is_file():
            raise SystemExit(f"Experiment results file not found: {experiment_path}")
        experiment_results = experiment_path.read_text(encoding="utf-8")
        experiment_results_source = "file"
        experiment_results_path = str(experiment_path)
        result_preflight_path = experiment_path
    else:
        dataset_csv_dir = code_path / "dataset_csv"
        experiment_results = _build_tcga_cohort_summary(dataset_csv_dir)
        experiment_results_source = "tcga_cohort_csv"
        experiment_results_path = str(dataset_csv_dir)
        result_preflight_path = dataset_csv_dir

    result_preflight = _validate_results_text(
        result_preflight_path,
        experiment_results,
        source=experiment_results_source,
        **_experiment_contract_kwargs(args),
        **_experiment_provenance_kwargs(args),
        **_experiment_artifact_consistency_kwargs(args),
    )
    if args.strict_results and not _validated_results_are_strictly_acceptable(result_preflight):
        raise SystemExit("Sample failed: experiment result validation failed in strict mode.")

    output_dir = Path(args.output_dir)
    project_name = args.project_name or _default_project_name(output_dir)
    request = PaperRequest(
        project_name=project_name,
        target_venue="TPAMI",
        baseline_pdf_path=str(baseline_pdf),
        code_path=str(code_path),
        method_notes=(
            "Hyper-ProtoSurv explores adaptive hypergraph prototype learning, "
            "bidirectional hyperedge updates, cross-attention fusion, and reconstruction "
            "regularization as reflected by the code structure and available TCGA cohort metadata."
        ),
        experiment_results=experiment_results,
        keywords=[
            "whole-slide images",
            "survival prediction",
            "computational pathology",
            "hypergraph learning",
        ],
        skip_llm_self_review=not args.allow_llm or args.skip_llm_self_review,
    )
    state = PaperWorkflow().run(request)
    _record_runtime_modes(
        state,
        network_mode=network_mode,
        llm_mode=llm_mode,
        compile_latex_requested=compile_latex_requested,
        llm_config=runtime_llm_config,
    )
    state.setdefault("artifacts", {})["experiment_results_source"] = experiment_results_source
    state["artifacts"]["experiment_results_path"] = experiment_results_path
    _record_result_preflight(state, result_preflight)
    SubmissionReadinessAgent().run(state)
    DraftReportAgent().run(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "draft.md"
    markdown_path.write_text(state["final_markdown"], encoding="utf-8")
    print(f"Markdown written to {markdown_path}")

    if args.zip:
        zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
        print(f"Overleaf zip written to {zip_path}")

    summary_path, acceptance_report_path = _write_run_reports(
        state,
        summary_path=output_dir / "RUN_SUMMARY.json",
        markdown_path=markdown_path,
        acceptance_report_path=Path(args.acceptance_report) if args.acceptance_report else None,
        min_llm_sections=0,
    )
    print(f"Run summary written to {summary_path}")
    print(f"Acceptance report written to {acceptance_report_path}")
    print(f"Review findings: {len(state.get('review_findings', []))}")
    print(f"Template source: {state['venue_template'].template_source}")
    print(f"Bibliography entries: {len(state.get('bibliography', []))}")
    print(f"Network mode: {network_mode}")
    print(f"LLM mode: {llm_mode}")
    print(f"LLM self-review: {_llm_self_review_mode(state)}")


def _run_tcga_doctor(args: argparse.Namespace) -> None:
    example_root = Path(args.example_root)
    experiment_path = _tcga_result_path(args, example_root)
    blocking: list[str] = []
    checks: list[dict[str, object]] = []
    result_summary = None

    print("TCGA project doctor:")
    checks.append(_print_doctor_check("Example root", example_root.exists(), str(example_root)))
    if not example_root.exists():
        blocking.append(f"Example root not found: {example_root}")

    baseline_pdf = None
    try:
        baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
        checks.append(_print_doctor_check("Baseline PDF", True, str(baseline_pdf)))
    except SystemExit as exc:
        checks.append(_print_doctor_check("Baseline PDF", False, str(example_root / "baseline")))
        blocking.append(str(exc))

    code_path = example_root / "code" / "hyper-protosurv"
    code_ok = code_path.is_dir()
    checks.append(_print_doctor_check("Code path", code_ok, str(code_path)))
    if not code_ok:
        blocking.append(f"Hyper-ProtoSurv code directory not found: {code_path}")

    if not experiment_path.is_file():
        checks.append(_print_doctor_check("Experiment results", False, str(experiment_path)))
        if args.write_template:
            experiment_path.parent.mkdir(parents=True, exist_ok=True)
            experiment_path.write_text(experiment_results_template(), encoding="utf-8")
            print(f"- Result template written: {experiment_path}")
            blocking.append(f"Fill every TODO in generated result template: {experiment_path}")
        else:
            blocking.append(
                "TCGA experiment results file missing. Create one with "
                f"`paper-agent tcga-doctor --example-root {example_root} --write-template`."
            )
    else:
        checks.append(_print_doctor_check("Experiment results", True, str(experiment_path)))
        result_summary = _validate_results_file(
            experiment_path,
            **_experiment_contract_kwargs(args),
            **_experiment_quality_kwargs(args, tcga_defaults=True),
            require_provenance=bool(args.require_provenance or args.submission_grade),
            require_artifact_consistency=bool(args.require_artifact_consistency or args.submission_grade),
        )
        if not _validated_results_are_strictly_acceptable(result_summary):
            blocking.append("Experiment result validation is not strict-acceptable.")

    llm_config = load_llm_config()
    checks.append(_print_doctor_check("LLM static config", llm_config.configured, _llm_config_label(llm_config)))
    if args.submission_grade and not llm_config.configured:
        blocking.append("Submission-grade TCGA generation requires a configured LLM.")
    llm_live_preflight: dict[str, object] = {"status": "skipped"}
    if args.live_llm and llm_config.configured:
        try:
            live_result = _llm_preflight_check(LLMClient(llm_config), llm_config, context="TCGA doctor")
            llm_live_preflight = {"status": "pass", **live_result}
            checks.append(_print_doctor_check("LLM live preflight", True, _llm_config_label(llm_config)))
        except SystemExit as exc:
            llm_live_preflight = {
                "status": "fail",
                "diagnostics": _llm_failure_diagnostics(llm_config, str(exc)),
            }
            checks.append(_print_doctor_check("LLM live preflight", False, _llm_config_label(llm_config)))
            blocking.append(str(exc))
    elif args.live_llm:
        llm_live_preflight = {
            "status": "fail",
            "diagnostics": _llm_failure_diagnostics(llm_config, "LLM is not configured"),
        }
        checks.append(_print_doctor_check("LLM live preflight", False, "not configured"))
        blocking.append("Cannot run LLM live preflight because LLM is not configured.")
    else:
        print("- LLM live preflight: SKIP (pass --live-llm to call the provider)")
        checks.append(
            {
                "name": "LLM live preflight",
                "status": "SKIP",
                "detail": "pass --live-llm to call the provider",
            }
        )

    latex_status = _latex_toolchain_status()
    checks.append(
        _print_doctor_check(
            "LaTeX compiler",
            bool(latex_status["available"]),
            str(latex_status.get("preferred_tool") or latex_status.get("install_hint")),
        )
    )
    if args.submission_grade and not latex_status["available"]:
        blocking.append(f"LaTeX compiler missing. Install hint: {latex_status['install_hint']}")

    next_actions = _tcga_doctor_next_actions(
        args,
        example_root,
        experiment_path,
        result_summary,
        llm_live_preflight,
        latex_status,
    )
    summary = {
        "status": "fail" if blocking else "pass",
        "submission_grade": bool(args.submission_grade),
        "example_root": str(example_root),
        "experiment_results": str(experiment_path),
        "checks": checks,
        "blocking_items": blocking,
        "result_validation": result_summary,
        "llm": _llm_static_summary(llm_config),
        "llm_live_preflight": llm_live_preflight,
        "latex": latex_status,
        "next_actions": next_actions,
    }
    if next_actions:
        summary["next_action"] = next_actions[0]["next_action"]
        summary["next_command"] = next_actions[0]["next_command"]
    if getattr(args, "summary", ""):
        summary_path = _write_run_summary_data(summary, Path(args.summary))
        print(f"TCGA doctor summary written to {summary_path}")

    if blocking:
        print("Overall: FAIL")
        print("Blocking items:")
        for item in blocking:
            print(f"- {item}")
        if next_actions:
            print("Next actions:")
            for action in next_actions:
                print(f"- {action['phase']}: {action['next_action']}")
                print(f"  Command: {action['next_command']}")
        raise SystemExit("TCGA doctor failed: fix blocking items before running tcga-draft.")
    print("Overall: PASS")
    if args.submission_grade:
        print("Ready command: paper-agent tcga-draft --submission-grade")
    else:
        print("Ready command: paper-agent tcga-draft")


def _tcga_doctor_next_actions(
    args: argparse.Namespace,
    example_root: Path,
    experiment_path: Path,
    result_summary: dict | None,
    llm_live_preflight: dict[str, object],
    latex_status: dict[str, object],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    if not experiment_path.is_file():
        actions.append(
            {
                "phase": "missing_result_file",
                "next_action": "Create result CSV artifact templates, fill every TODO with real trained-model outputs, then generate the paper-facing Markdown.",
                "next_command": _tcga_artifact_template_command(args, example_root),
                "markdown_template_command": _tcga_doctor_write_template_command(args, example_root),
                "results_from_artifacts_command": _tcga_results_from_artifacts_command(args, example_root),
                "validation_command": _validate_results_command(args, example_root),
            }
        )
    elif result_summary is None and _file_contains_todo(experiment_path):
        actions.append(_tcga_doctor_result_repair_action(args, example_root, experiment_path, None, has_todos=True))
    elif result_summary is not None and not _validated_results_are_strictly_acceptable(result_summary):
        actions.append(
            _tcga_doctor_result_repair_action(
                args,
                example_root,
                experiment_path,
                result_summary,
                has_todos=_file_contains_todo(experiment_path),
            )
        )

    if llm_live_preflight.get("status") == "fail":
        diagnostics = llm_live_preflight.get("diagnostics", {})
        diagnosis = (
            diagnostics.get("diagnosis")
            if isinstance(diagnostics, dict)
            else "Fix LLM configuration or provider availability."
        )
        actions.append(
            {
                "phase": "llm_live_preflight",
                "next_action": f"{diagnosis} Then rerun tcga-doctor with --live-llm.",
                "next_command": "paper-agent llm-doctor --summary outputs\\llm-doctor.json",
                "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
            }
        )

    if bool(getattr(args, "submission_grade", False)) and not bool(latex_status.get("available")):
        actions.append(
            {
                "phase": "latex_compiler",
                "next_action": "Install a local LaTeX compiler before submission-grade drafting.",
                "next_command": str(latex_status.get("install_hint") or "conda install -n agent -c conda-forge tectonic"),
            }
        )
    return actions


def _tcga_doctor_result_repair_action(
    args: argparse.Namespace,
    example_root: Path,
    experiment_path: Path,
    result_summary: dict | None,
    *,
    has_todos: bool,
) -> dict[str, object]:
    next_action = (
        "Replace every TODO placeholder with real trained-model outputs or regenerate the result file from completed CSV artifacts, then validate strictly."
        if has_todos
        else "Repair the result file until contract, quality, provenance, and artifact-consistency checks are strict-acceptable."
    )
    action: dict[str, object] = {
        "phase": "result_validation",
        "next_action": next_action,
        "next_command": _tcga_artifact_template_command(args, example_root) if has_todos else _validate_results_command(args, example_root),
        "artifact_template_command": _tcga_artifact_template_command(args, example_root),
        "results_from_artifacts_command": _tcga_results_from_artifacts_command(args, example_root),
        "validation_command": _validate_results_command(args, example_root),
        "experiment_results": str(experiment_path),
        "has_todo_placeholders": has_todos,
    }
    if result_summary:
        action["diagnostics"] = _tcga_result_validation_diagnostics(result_summary)
    return action


def _tcga_result_validation_diagnostics(result_summary: dict) -> dict[str, object]:
    contract = result_summary.get("experiment_contract", {})
    quality = result_summary.get("experiment_quality", {})
    provenance = result_summary.get("experiment_provenance", {})
    consistency = result_summary.get("experiment_artifact_consistency", {})
    return {
        "contract_status": contract.get("status", "unknown"),
        "contract_errors": contract.get("errors", []),
        "quality_status": quality.get("status", "unknown"),
        "quality_errors": quality.get("errors", []),
        "provenance_status": provenance.get("status", "unknown"),
        "provenance_errors": provenance.get("errors", []),
        "artifact_consistency_status": consistency.get("status", "unknown"),
        "artifact_consistency_errors": consistency.get("errors", []),
    }


def _file_contains_todo(path: Path) -> bool:
    if not path.is_file():
        return False
    return bool(re.search(r"\bTODO\b", path.read_text(encoding="utf-8", errors="ignore"), flags=re.IGNORECASE))


def _tcga_doctor_write_template_command(args: argparse.Namespace, example_root: Path) -> str:
    parts = ["paper-agent", "tcga-doctor", "--example-root", str(example_root), "--write-template"]
    if getattr(args, "experiment_results", ""):
        parts.extend(["--experiment-results", args.experiment_results])
    return " ".join(_powershell_arg(part) for part in parts)


def _run_tcga_results_from_artifacts(args: argparse.Namespace) -> dict | None:
    example_root = Path(args.example_root)
    output_path = _resolve_project_relative_path(args.output) if args.output else example_root / "results" / "tcga_results.md"
    artifact_dir = _tcga_artifact_dir(args, example_root)
    detected = (
        _detect_tcga_artifact_csvs(
            artifact_dir,
            method=args.method,
            baseline=args.baseline,
            metric=args.metric,
        )
        if artifact_dir
        else {}
    )
    main_csv = _resolve_required_or_detected_file(args.main_csv, detected.get("main"), "Main result CSV")
    ablation_csv = _resolve_optional_or_detected_file(args.ablation_csv, detected.get("ablation"), "Ablation CSV")
    sensitivity_csv = _resolve_optional_or_detected_file(args.sensitivity_csv, detected.get("sensitivity"), "Sensitivity CSV")
    stats_csv = _resolve_optional_or_detected_file(args.stats_csv, detected.get("stats"), "Statistical-test CSV")

    main_rows = _read_csv_dicts(main_csv)
    datasets = args.dataset or _detect_result_datasets(main_rows, metric=args.metric)
    if not datasets:
        raise SystemExit("No datasets detected. Pass --dataset or use wide headers such as `BLCA C-index`.")

    main_table = _tcga_main_result_table(
        main_rows,
        datasets=datasets,
        metric=args.metric,
        method=args.method,
        baseline=args.baseline,
    )
    ablation_table = (
        _tcga_single_value_table(
            _read_csv_dicts(ablation_csv),
            label_column_candidates=("variant", "method"),
            value_column_hint=args.metric,
            table_label="Variant",
            dataset_filter="Average",
            allowed_label_patterns=(args.method, r"\b(w/o|without|no|ablat(?:e|ed|ion)|remove(?:d)?|minus)\b"),
        )
        if ablation_csv
        else []
    )
    sensitivity_table = (
        _tcga_single_value_table(
            _read_csv_dicts(sensitivity_csv),
            label_column_candidates=("lambda_rec", "parameter_value", "value_name"),
            value_column_hint=args.metric,
            table_label="lambda_rec",
            dataset_filter="Average",
        )
        if sensitivity_csv
        else []
    )
    stats_table = _tcga_stats_table(_read_csv_dicts(stats_csv)) if stats_csv else []
    provenance_sources = _tcga_provenance_sources(
        main_csv=main_csv,
        ablation_csv=ablation_csv,
        sensitivity_csv=sensitivity_csv,
        stats_csv=stats_csv,
    )
    markdown = _render_tcga_results_markdown(
        metric=args.metric,
        datasets=datasets,
        main_table=main_table,
        ablation_table=ablation_table,
        sensitivity_table=sensitivity_table,
        stats_table=stats_table,
        provenance_sources=provenance_sources,
        output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"TCGA result file written to {output_path}")
    print(f"Main datasets: {', '.join(datasets)}")
    print(f"Provenance artifacts: {len(provenance_sources)}")
    if detected:
        _print_detected_tcga_artifacts(detected)

    if args.strict:
        summary = _validate_results_file(
            output_path,
            **_experiment_contract_kwargs(args),
            expected_datasets=list(datasets),
            expected_metrics=[args.metric],
            expected_method=args.method,
            expected_baseline=args.baseline,
            require_provenance=True,
            require_artifact_consistency=True,
        )
        if not _validated_results_are_strictly_acceptable(summary):
            raise SystemExit("Generated TCGA result file failed strict validation.")
        return summary
    return None


def _run_tcga_export_artifacts(args: argparse.Namespace) -> None:
    input_csv = _resolve_required_file(args.input_csv, "Input export CSV")
    output_dir = _resolve_project_relative_path(args.output_dir)
    rows = _read_csv_dicts(input_csv)
    try:
        written = write_tcga_artifact_exports_from_rows(
            output_dir,
            rows,
            role_column=args.role_column,
            method=args.method,
            baseline=args.baseline,
            metric=args.metric,
            seed=args.seed,
            force=bool(args.force),
            require_complete=not bool(args.allow_partial),
        )
    except (FileExistsError, TypeError, ValueError) as exc:
        raise SystemExit(f"TCGA artifact export failed: {exc}") from exc

    print(f"TCGA artifact exports written to {output_dir}")
    for name in sorted(written):
        print(f"- {name}: {written[name]}")
    print(f"Ready command: paper-agent tcga-artifacts-doctor --artifacts-dir {output_dir} --summary {output_dir / 'artifact-doctor.json'}")
    print(f"Next command: paper-agent tcga-results-from-artifacts --artifacts-dir {output_dir} --strict")


def _run_tcga_demo_artifact_flow(args: argparse.Namespace) -> None:
    input_csv = _resolve_required_file(args.input_csv, "Demo training summary CSV")
    output_dir = _resolve_project_relative_path(args.output_dir)
    artifacts_dir = output_dir / "artifacts"
    result_path = output_dir / "tcga_results.md"
    rows = _read_csv_dicts(input_csv)
    try:
        written = write_tcga_artifact_exports_from_rows(
            artifacts_dir,
            rows,
            method=args.method,
            baseline=args.baseline,
            metric=args.metric,
            seed=args.seed,
            force=bool(args.force),
        )
    except (FileExistsError, TypeError, ValueError) as exc:
        raise SystemExit(f"TCGA demo artifact flow failed: {exc}") from exc

    print(f"TCGA demo artifacts written to {artifacts_dir}")
    print(f"Artifact files: {len(written)}")
    result_args = argparse.Namespace(
        example_root=str(output_dir),
        artifacts_dir=str(artifacts_dir),
        main_csv="",
        ablation_csv="",
        sensitivity_csv="",
        stats_csv="",
        output=str(result_path),
        method=args.method,
        baseline=args.baseline,
        metric=args.metric,
        dataset=[],
        strict=True,
        require_ablation=True,
        require_sensitivity=True,
        require_statistical_tests=True,
    )
    result_summary = _run_tcga_results_from_artifacts(result_args) or {}
    summary_path = _resolve_project_relative_path(args.summary) if args.summary else output_dir / "RUN_SUMMARY.json"
    run_summary = _tcga_demo_artifact_flow_summary(
        args,
        input_csv,
        output_dir,
        artifacts_dir,
        result_path,
        summary_path,
        written,
        result_summary,
    )
    written_summary = _write_run_summary_data(run_summary, summary_path)
    print(f"TCGA demo result Markdown written to {result_path}")
    print(f"TCGA demo summary written to {written_summary}")
    print(f"Ready command: paper-agent validate-results --experiment-results {result_path} --strict")


def _tcga_demo_artifact_flow_summary(
    args: argparse.Namespace,
    input_csv: Path,
    output_dir: Path,
    artifacts_dir: Path,
    result_path: Path,
    summary_path: Path,
    written_artifacts: dict[str, Path],
    result_summary: dict,
) -> dict[str, object]:
    contract = result_summary.get("experiment_contract", {})
    quality = result_summary.get("experiment_quality", {})
    provenance = result_summary.get("experiment_provenance", {})
    consistency = result_summary.get("experiment_artifact_consistency", {})
    return {
        "status": "pass" if _validated_results_are_strictly_acceptable(result_summary) else "fail",
        "pipeline_phase": "tcga_artifact_flow_demo",
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "artifacts_dir": str(artifacts_dir),
        "experiment_results": str(result_path),
        "artifact_files": {name: str(path) for name, path in sorted(written_artifacts.items())},
        "artifact_schema_path": str(written_artifacts.get("ARTIFACT_SCHEMA.json", artifacts_dir / "ARTIFACT_SCHEMA.json")),
        "result_validation": result_summary,
        "experiment_contract_status": contract.get("status", "unknown"),
        "experiment_quality_status": quality.get("status", "unknown"),
        "experiment_provenance_status": provenance.get("status", "unknown"),
        "experiment_artifact_consistency_status": consistency.get("status", "unknown"),
        "artifact_consistency_matched": consistency.get("checks", {}).get("matched_values", 0),
        "artifact_consistency_missing": consistency.get("checks", {}).get("missing_values", 0),
        "artifact_consistency_mismatched": consistency.get("checks", {}).get("mismatched_values", 0),
        "method": args.method,
        "baseline": args.baseline,
        "metric": args.metric,
        "seed": args.seed,
        "next_command": f"paper-agent validate-results --experiment-results {_powershell_arg(result_path)} --strict",
        "draft_command": f"paper-agent tcga-draft --artifact-flow-summary {_powershell_arg(summary_path)}",
        "note": "Bundled demo values are for local workflow validation only, not evidence for a real paper.",
    }


def _run_tcga_demo_paper(args: argparse.Namespace) -> None:
    output_dir = _resolve_project_relative_path(args.output_dir)
    artifact_flow_dir = output_dir / "artifact-flow"
    draft_dir = output_dir / "draft"
    artifact_summary_path = artifact_flow_dir / "RUN_SUMMARY.json"
    draft_summary_path = draft_dir / "RUN_SUMMARY.json"
    zip_path = _resolve_project_relative_path(args.zip) if args.zip else output_dir / "paper-overleaf.zip"

    print("TCGA demo paper: generating artifact flow")
    flow_args = argparse.Namespace(
        input_csv=args.input_csv,
        output_dir=str(artifact_flow_dir),
        method=args.method,
        baseline=args.baseline,
        metric=args.metric,
        seed=args.seed,
        force=bool(args.force),
        summary=str(artifact_summary_path),
    )
    _run_tcga_demo_artifact_flow(flow_args)

    print("TCGA demo paper: drafting paper package")
    project_name = args.project_name or f"{_default_project_name(output_dir)}-draft"
    draft_args = argparse.Namespace(
        example_root=args.example_root,
        experiment_results="",
        artifact_flow_summary=str(artifact_summary_path),
        project_name=project_name,
        target_venue=args.target_venue,
        output_dir=str(draft_dir),
        zip=str(zip_path),
        template_zip=args.template_zip,
        template_dir=args.template_dir,
        online=bool(getattr(args, "online", False)),
        offline=bool(getattr(args, "offline", False)),
        disable_llm=not bool(args.use_llm),
        min_llm_sections=args.min_llm_sections,
        skip_llm_self_review=bool(args.skip_llm_self_review),
        compile_latex=bool(args.compile_latex),
        submission_grade=False,
        keyword=list(args.keyword or []),
        require_ablation=bool(args.require_ablation),
        require_sensitivity=bool(args.require_sensitivity),
        require_statistical_tests=bool(args.require_statistical_tests),
        expected_dataset=list(args.expected_dataset or []),
        expected_metric=list(args.expected_metric or []),
        expected_method=args.expected_method,
        expected_baseline=args.expected_baseline,
        require_provenance=bool(args.require_provenance),
        require_artifact_consistency=bool(args.require_artifact_consistency),
    )
    _run_tcga_draft(draft_args)

    artifact_summary = _read_json_object(artifact_summary_path, "TCGA artifact-flow summary")
    draft_summary = _read_json_object(draft_summary_path, "TCGA draft summary")
    demo_summary = _tcga_demo_paper_summary(
        args,
        output_dir,
        artifact_flow_dir,
        draft_dir,
        artifact_summary_path,
        draft_summary_path,
        zip_path,
        artifact_summary,
        draft_summary,
    )
    summary_path = _write_run_summary_data(demo_summary, output_dir / "RUN_SUMMARY.json")
    print(f"TCGA demo paper summary written to {summary_path}")
    print("TCGA demo paper completed.")


def _tcga_demo_paper_summary(
    args: argparse.Namespace,
    output_dir: Path,
    artifact_flow_dir: Path,
    draft_dir: Path,
    artifact_summary_path: Path,
    draft_summary_path: Path,
    zip_path: Path,
    artifact_summary: dict,
    draft_summary: dict,
) -> dict[str, object]:
    draft_outputs = draft_summary.get("outputs", {}) if isinstance(draft_summary.get("outputs", {}), dict) else {}
    return {
        "status": "pass",
        "pipeline_phase": "tcga_demo_paper",
        "target_venue": args.target_venue,
        "project_name": args.project_name or f"{_default_project_name(output_dir)}-draft",
        "output_dir": str(output_dir),
        "artifact_flow_dir": str(artifact_flow_dir),
        "draft_dir": str(draft_dir),
        "artifact_flow_summary_path": str(artifact_summary_path),
        "draft_summary_path": str(draft_summary_path),
        "experiment_results": artifact_summary.get("experiment_results", ""),
        "artifact_flow_status": artifact_summary.get("status", "unknown"),
        "draft_experiment_contract_status": draft_summary.get("experiment_contract_status", "unknown"),
        "draft_experiment_quality_status": draft_summary.get("experiment_quality_status", "unknown"),
        "draft_experiment_provenance_status": draft_summary.get("experiment_provenance_status", "unknown"),
        "draft_experiment_artifact_consistency_status": draft_summary.get(
            "experiment_artifact_consistency_status",
            "unknown",
        ),
        "llm_mode": draft_summary.get("inputs", {}).get("llm_mode", "unknown")
        if isinstance(draft_summary.get("inputs", {}), dict)
        else "unknown",
        "outputs": {
            "markdown": draft_outputs.get("markdown", ""),
            "latex_project_dir": draft_outputs.get("latex_project_dir", ""),
            "latex_output_path": draft_outputs.get("latex_output_path", ""),
            "latex_zip_path": str(zip_path),
            "draft_report_path": draft_outputs.get("draft_report_path", ""),
            "acceptance_report_path": draft_outputs.get("acceptance_report_path", ""),
        },
        "note": "Bundled TCGA demo values are for local workflow validation only, not evidence for a real paper.",
    }


def _run_tcga_artifacts_doctor(args: argparse.Namespace) -> None:
    example_root = Path(args.example_root)
    artifact_dir = _tcga_artifact_dir(args, example_root)
    explicit_paths = {
        "main": _resolve_optional_file(args.main_csv, "Main result CSV"),
        "ablation": _resolve_optional_file(args.ablation_csv, "Ablation CSV"),
        "sensitivity": _resolve_optional_file(args.sensitivity_csv, "Sensitivity CSV"),
        "stats": _resolve_optional_file(args.stats_csv, "Statistical-test CSV"),
    }
    detected = (
        _detect_tcga_artifact_csvs(
            artifact_dir,
            method=args.method,
            baseline=args.baseline,
            metric=args.metric,
        )
        if artifact_dir
        else {}
    )
    role_paths = {
        role: explicit_paths.get(role) or detected.get(role)
        for role in ("main", "ablation", "sensitivity", "stats")
    }
    required_roles = {
        "main": True,
        "ablation": bool(args.require_ablation),
        "sensitivity": bool(args.require_sensitivity),
        "stats": bool(args.require_statistical_tests),
    }

    print("TCGA artifact doctor:")
    has_input = bool(artifact_dir or any(explicit_paths.values()))
    checks: list[dict[str, object]] = []
    role_summaries: dict[str, dict[str, object]] = {}
    checks.append(_print_doctor_check("Artifact directory", has_input, str(artifact_dir or "not provided")))
    if detected:
        _print_detected_tcga_artifacts(detected)

    blocking: list[str] = []
    datasets = list(args.dataset or [])
    for role in ("main", "ablation", "sensitivity", "stats"):
        label = _tcga_artifact_role_label(role)
        path = role_paths.get(role)
        required = required_roles[role]
        if not path:
            detail = f"missing; expected {_tcga_expected_artifact_schema(role)}"
            status_ok = not required
            checks.append(_print_doctor_check(label, status_ok, detail))
            role_summaries[role] = {
                "role": role,
                "label": label,
                "required": required,
                "status": "pass" if status_ok else "fail",
                "path": "",
                "detail": detail,
                "issues": [detail] if required else [],
                "datasets": [],
            }
            if required:
                blocking.append(f"Missing required {label}: {_tcga_expected_artifact_schema(role)}")
            continue
        diagnostic = _tcga_artifact_role_diagnostic(
            role,
            path,
            args=args,
            datasets=datasets,
        )
        status_ok = bool(diagnostic["ok"])
        checks.append(_print_doctor_check(label, status_ok, str(diagnostic["detail"])))
        for issue in diagnostic["issues"]:
            print(f"  - {issue}")
        role_summaries[role] = {
            "role": role,
            "label": label,
            "required": required,
            "status": "pass" if status_ok else "fail",
            "path": str(path),
            "detail": str(diagnostic["detail"]),
            "issues": list(diagnostic["issues"]),
            "datasets": list(diagnostic["datasets"]),
        }
        if role == "main" and diagnostic["datasets"]:
            datasets = list(diagnostic["datasets"])
        if not diagnostic["ok"]:
            blocking.append(f"{label} is not parseable: {path}")

    summary = _tcga_artifacts_doctor_summary(
        args,
        example_root,
        artifact_dir,
        detected,
        checks,
        role_summaries,
        blocking,
        datasets,
    )
    if getattr(args, "summary", ""):
        summary_path = _write_run_summary_data(summary, Path(args.summary))
        print(f"TCGA artifact doctor summary written to {summary_path}")

    if blocking:
        print("Overall: FAIL")
        print("Blocking items:")
        for item in blocking:
            print(f"- {item}")
        raise SystemExit("TCGA artifact doctor failed: fix CSV artifacts before generating tcga_results.md.")
    print("Overall: PASS")
    if artifact_dir:
        print(f"Ready command: paper-agent tcga-results-from-artifacts --artifacts-dir {artifact_dir} --strict")
    else:
        print("Ready command: paper-agent tcga-results-from-artifacts --main-csv <path> --strict")


def _tcga_artifacts_doctor_summary(
    args: argparse.Namespace,
    example_root: Path,
    artifact_dir: Path | None,
    detected: dict[str, Path],
    checks: list[dict[str, object]],
    role_summaries: dict[str, dict[str, object]],
    blocking: list[str],
    datasets: list[str],
) -> dict[str, object]:
    status = "fail" if blocking else "pass"
    next_command = (
        _tcga_artifact_template_command(args, example_root)
        if blocking
        else _tcga_results_from_artifacts_command(args, example_root)
    )
    next_action = (
        "Export or fill the required TCGA CSV artifacts, then rerun tcga-artifacts-doctor."
        if blocking
        else "Generate the paper-facing TCGA result Markdown from the validated CSV artifacts."
    )
    return {
        "status": status,
        "example_root": str(example_root),
        "artifact_dir": str(artifact_dir or ""),
        "detected_artifacts": {role: str(path) for role, path in detected.items()},
        "checks": checks,
        "roles": role_summaries,
        "blocking_items": blocking,
        "datasets": datasets,
        "expected_schemas": {role: _tcga_expected_artifact_schema(role) for role in ("main", "ablation", "sensitivity", "stats")},
        "next_action": next_action,
        "next_command": next_command,
        "results_from_artifacts_command": _tcga_results_from_artifacts_command(args, example_root),
        "artifact_template_command": _tcga_artifact_template_command(args, example_root),
    }


def _run_tcga_artifact_template(args: argparse.Namespace) -> None:
    output_dir = _resolve_project_relative_path(args.output_dir)
    datasets = list(args.dataset or _tcga_default_datasets())
    files = _tcga_artifact_template_bundle(
        style=args.style,
        datasets=datasets,
        method=args.method,
        baseline=args.baseline,
        metric=args.metric,
        seed=args.seed,
    )
    written_paths = _write_tcga_artifact_template_bundle(output_dir, files, force=bool(args.force))
    for path in written_paths:
        print(f"- Wrote {path}")
    print(f"TCGA artifact export templates written to {output_dir}")
    print("Replace every TODO with real trained-model outputs before running tcga-artifacts-doctor.")


def _write_tcga_artifact_template_bundle(
    output_dir: Path,
    files: dict[str, str],
    *,
    force: bool = False,
) -> list[Path]:
    conflicts = [name for name in files if (output_dir / name).exists()]
    if conflicts and not force:
        raise SystemExit(
            "Refusing to overwrite existing TCGA artifact template files: "
            + ", ".join(conflicts)
            + ". Pass --force to overwrite."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for name, content in files.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written_paths.append(path)
    return written_paths


def _run_tcga_preflight(args: argparse.Namespace) -> None:
    checks: list[dict[str, object]] = []
    blocking: list[str] = []
    submission_grade = bool(args.submission_grade)

    print("TCGA preflight:")
    if submission_grade and args.offline:
        _add_preflight_check(
            checks,
            "Network mode",
            "FAIL",
            "submission-grade requires online mode; remove --offline",
            blocking,
        )
    else:
        network_mode = "online" if args.online or submission_grade else "offline" if args.offline else "environment/default"
        _add_preflight_check(checks, "Network mode", "PASS", network_mode, blocking, blocking_item=False)

    example_root = Path(args.example_root)
    experiment_path = _tcga_result_path(args, example_root)
    _add_preflight_check(
        checks,
        "Example root",
        "PASS" if example_root.exists() else "FAIL",
        str(example_root),
        blocking,
        blocking_item=not example_root.exists(),
    )
    try:
        baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
        _add_preflight_check(checks, "Baseline PDF", "PASS", str(baseline_pdf), blocking, blocking_item=False)
    except SystemExit as exc:
        _add_preflight_check(checks, "Baseline PDF", "FAIL", str(exc), blocking)

    code_path = example_root / "code" / "hyper-protosurv"
    _add_preflight_check(
        checks,
        "Code path",
        "PASS" if code_path.is_dir() else "FAIL",
        str(code_path),
        blocking,
        blocking_item=not code_path.is_dir(),
    )

    artifact_ready = _record_tcga_artifact_preflight(args, example_root, checks, blocking)

    require_provenance = bool(args.require_provenance or submission_grade)
    require_artifact_consistency = bool(args.require_artifact_consistency or submission_grade)
    result_summary = None
    if experiment_path.is_file():
        result_summary = _validate_results_file(
            experiment_path,
            **_experiment_contract_kwargs(args),
            **_experiment_quality_kwargs(args, tcga_defaults=True),
            require_provenance=require_provenance,
            require_artifact_consistency=require_artifact_consistency,
        )
        result_ok = _validated_results_are_strictly_acceptable(result_summary)
        _add_preflight_check(
            checks,
            "Experiment results",
            "PASS" if result_ok else "FAIL",
            (
                f"{experiment_path}; contract={result_summary.get('experiment_contract', {}).get('status', 'unknown')}; "
                f"quality={result_summary.get('experiment_quality', {}).get('status', 'unknown')}; "
                f"provenance={result_summary.get('experiment_provenance', {}).get('status', 'unknown')}; "
                f"artifact_consistency={result_summary.get('experiment_artifact_consistency', {}).get('status', 'unknown')}"
            ),
            blocking,
            blocking_item=not result_ok,
        )
    else:
        status = "WARN" if artifact_ready else "FAIL"
        detail = (
            f"{experiment_path} missing; tcga-pipeline can generate it from detected artifacts"
            if artifact_ready
            else (
                f"{experiment_path} missing and no complete artifact set is available. "
                f"Initialize real-result CSV templates with: {_tcga_artifact_template_command(args, example_root)}"
            )
        )
        _add_preflight_check(checks, "Experiment results", status, detail, blocking, blocking_item=not artifact_ready)

    llm_config = load_llm_config()
    llm_required = bool(submission_grade or not args.disable_llm)
    llm_live_preflight: dict[str, object] = {"status": "skipped"}
    if args.disable_llm and not submission_grade:
        _add_preflight_check(checks, "LLM static config", "PASS", "disabled by --disable-llm", blocking, blocking_item=False)
    else:
        _add_preflight_check(
            checks,
            "LLM static config",
            "PASS" if llm_config.configured else "FAIL",
            _llm_config_label(llm_config),
            blocking,
            blocking_item=llm_required and not llm_config.configured,
        )
    if args.live_llm:
        if not llm_config.configured:
            llm_live_preflight = {
                "status": "fail",
                "diagnostics": _llm_failure_diagnostics(llm_config, "LLM is not configured"),
            }
            _add_preflight_check(checks, "LLM live preflight", "FAIL", "LLM is not configured", blocking)
        else:
            try:
                live_result = _llm_preflight_check(LLMClient(llm_config), llm_config, context="TCGA preflight")
                llm_live_preflight = {"status": "pass", **live_result}
                _add_preflight_check(checks, "LLM live preflight", "PASS", _llm_config_label(llm_config), blocking, blocking_item=False)
            except SystemExit as exc:
                llm_live_preflight = {
                    "status": "fail",
                    "diagnostics": _llm_failure_diagnostics(llm_config, str(exc)),
                }
                _add_preflight_check(checks, "LLM live preflight", "FAIL", str(exc), blocking)
    else:
        _add_preflight_check(checks, "LLM live preflight", "SKIP", "pass --live-llm to call the provider", blocking, blocking_item=False)

    latex_status = _latex_toolchain_status()
    latex_required = bool(args.compile_latex or submission_grade)
    _add_preflight_check(
        checks,
        "LaTeX compiler",
        "PASS" if latex_status["available"] else "FAIL" if latex_required else "WARN",
        str(latex_status.get("preferred_tool") or latex_status.get("install_hint")),
        blocking,
        blocking_item=latex_required and not latex_status["available"],
    )

    for check in checks:
        print(f"- {check['name']}: {check['status']} ({check['detail']})")
    overall = "FAIL" if blocking else "PASS"
    print(f"Overall: {overall}")
    if blocking:
        print("Blocking items:")
        for item in blocking:
            print(f"- {item}")
    else:
        print("Ready command: paper-agent tcga-pipeline --submission-grade" if submission_grade else "Ready command: paper-agent tcga-pipeline")

    readiness_contract = _tcga_preflight_readiness_contract(
        args,
        example_root,
        experiment_path,
        checks,
        blocking,
        artifact_ready=artifact_ready,
        result_summary=result_summary,
        llm_config=llm_config,
        llm_live_preflight=llm_live_preflight,
        latex_status=latex_status,
    )
    _print_tcga_preflight_readiness_contract(readiness_contract)

    summary = {
        "status": overall.lower(),
        "submission_grade": submission_grade,
        "example_root": str(example_root),
        "experiment_results": str(experiment_path),
        "checks": checks,
        "blocking_items": blocking,
        "readiness_contract": readiness_contract,
        "next_actions": readiness_contract["next_actions"],
        "result_validation": result_summary,
        "llm": _llm_static_summary(llm_config),
        "llm_live_preflight": llm_live_preflight,
    }
    if args.summary:
        summary_path = _write_run_summary_data(summary, Path(args.summary))
        print(f"Preflight summary written to {summary_path}")
    if blocking:
        raise SystemExit("TCGA preflight failed: fix blocking items before running tcga-pipeline.")


def _tcga_preflight_readiness_contract(
    args: argparse.Namespace,
    example_root: Path,
    experiment_path: Path,
    checks: list[dict[str, object]],
    blocking: list[str],
    *,
    artifact_ready: bool,
    result_summary: dict | None,
    llm_config: LLMConfig,
    llm_live_preflight: dict[str, object],
    latex_status: dict[str, object],
) -> dict[str, object]:
    submission_grade = bool(args.submission_grade)
    requirements: dict[str, dict[str, object]] = {}

    def add_requirement(
        category: str,
        status: str,
        detail: str,
        *,
        required: bool = True,
        next_action: str = "",
        command: str = "",
    ) -> None:
        requirements[category] = {
            "status": status,
            "required": required,
            "detail": detail,
            "next_action": next_action,
            "command": command,
        }

    network_check = _preflight_check(checks, "Network mode")
    network_failed = network_check.get("status") == "FAIL"
    add_requirement(
        "venue_network",
        "fail" if network_failed else "pass",
        str(network_check.get("detail", "")),
        required=submission_grade,
        next_action="Use --online or remove --offline for submission-grade runs." if network_failed else "",
    )

    baseline_check = _preflight_check(checks, "Baseline PDF")
    baseline_failed = baseline_check.get("status") == "FAIL"
    add_requirement(
        "baseline_pdf",
        "fail" if baseline_failed else "pass",
        str(baseline_check.get("detail", "")),
        next_action=f"Place the baseline PDF under {example_root / 'baseline'}." if baseline_failed else "",
    )

    code_check = _preflight_check(checks, "Code path")
    code_failed = code_check.get("status") == "FAIL"
    add_requirement(
        "code_path",
        "fail" if code_failed else "pass",
        str(code_check.get("detail", "")),
        next_action=f"Place the project code under {example_root / 'code' / 'hyper-protosurv'}." if code_failed else "",
    )

    artifact_checks = [
        check for check in checks if str(check.get("name", "")).startswith("Artifact ")
    ]
    artifact_failures = [check for check in artifact_checks if check.get("status") == "FAIL"]
    artifact_warnings = [check for check in artifact_checks if check.get("status") == "WARN"]
    artifact_status = "pass" if artifact_ready and not artifact_failures else "fail" if artifact_failures else "warn"
    add_requirement(
        "result_artifacts",
        artifact_status,
        _tcga_preflight_artifact_detail(artifact_checks),
        required=not bool(result_summary),
        next_action="Create or repair real-result CSV artifacts before drafting." if artifact_status == "fail" else "",
        command=_tcga_artifact_template_command(args, example_root) if artifact_status == "fail" else "",
    )

    result_check = _preflight_check(checks, "Experiment results")
    result_status_raw = str(result_check.get("status", "FAIL"))
    if result_status_raw == "PASS":
        result_status = "pass"
        result_next = ""
        result_command = ""
    elif result_status_raw == "WARN" and artifact_ready:
        result_status = "ready_to_generate"
        result_next = "Run tcga-pipeline to generate the paper-facing result Markdown from artifacts."
        result_command = _tcga_preflight_pipeline_command(args, example_root, experiment_path)
    else:
        result_status = "fail"
        result_next = "Provide a strict result Markdown file or create repairable result CSV artifacts."
        result_command = _tcga_artifact_template_command(args, example_root)
    add_requirement(
        "experiment_results",
        result_status,
        str(result_check.get("detail", "")),
        next_action=result_next,
        command=result_command,
    )

    llm_live_status = str(llm_live_preflight.get("status", "skipped"))
    llm_required = bool(submission_grade or not args.disable_llm)
    if args.disable_llm and not submission_grade:
        llm_status = "disabled"
        llm_next = ""
    elif not llm_config.configured and llm_required:
        llm_status = "fail"
        llm_next = "Configure an API key and TEXT_MODEL before LLM drafting."
    elif llm_live_status == "fail":
        llm_status = "fail"
        llm_next = "Fix provider quota/configuration or rerun without --live-llm for static-only preflight."
    else:
        llm_status = "pass"
        llm_next = ""
    add_requirement(
        "llm",
        llm_status,
        f"{_llm_config_label(llm_config)}; live_preflight={llm_live_status}",
        required=llm_required,
        next_action=llm_next,
        command="paper-agent llm-doctor" if llm_status == "fail" else "",
    )

    latex_required = bool(args.compile_latex or submission_grade)
    latex_available = bool(latex_status.get("available"))
    latex_status_value = "pass" if latex_available else "fail" if latex_required else "warn"
    add_requirement(
        "latex",
        latex_status_value,
        str(latex_status.get("preferred_tool") or latex_status.get("install_hint") or "not found"),
        required=latex_required,
        next_action="Install a local LaTeX compiler before submission-grade drafting." if latex_status_value == "fail" else "",
        command=str(latex_status.get("install_hint", "")) if latex_status_value == "fail" else "",
    )

    blocking_categories = [
        category for category, requirement in requirements.items() if requirement["status"] == "fail"
    ]
    next_actions = _tcga_preflight_next_actions(requirements)
    if not blocking_categories:
        next_actions.append(
            {
                "category": "pipeline",
                "action": "Run the TCGA pipeline with the checked inputs.",
                "command": _tcga_preflight_pipeline_command(args, example_root, experiment_path),
            }
        )
    return {
        "schema_version": TCGA_READINESS_CONTRACT_SCHEMA_VERSION,
        "status": "blocked" if blocking else "ready",
        "submission_grade": submission_grade,
        "ready_for_submission_grade": bool(submission_grade and not blocking),
        "ready_for_deterministic_draft": bool(not blocking),
        "blocking_categories": blocking_categories,
        "requirements": requirements,
        "next_actions": next_actions,
        "artifact_warnings": [str(check.get("detail", "")) for check in artifact_warnings],
    }


def _print_tcga_preflight_readiness_contract(contract: dict[str, object]) -> None:
    print("Readiness contract:")
    requirements = contract.get("requirements", {})
    if not isinstance(requirements, dict):
        return
    for category, requirement in requirements.items():
        if not isinstance(requirement, dict):
            continue
        print(f"- {category}: {requirement.get('status', 'unknown')} ({requirement.get('detail', '')})")


def _preflight_check(checks: list[dict[str, object]], name: str) -> dict[str, object]:
    for check in checks:
        if check.get("name") == name:
            return check
    return {"name": name, "status": "FAIL", "detail": "not recorded"}


def _tcga_preflight_artifact_detail(checks: list[dict[str, object]]) -> str:
    if not checks:
        return "no artifact checks were run"
    parts = []
    for check in checks:
        role = str(check.get("name", "Artifact")).replace("Artifact ", "")
        parts.append(f"{role}={check.get('status', 'unknown')}")
    return "; ".join(parts)


def _tcga_preflight_next_actions(requirements: dict[str, dict[str, object]]) -> list[dict[str, str]]:
    actions = []
    for category, requirement in requirements.items():
        if requirement.get("status") != "fail":
            continue
        action = str(requirement.get("next_action", "") or "")
        command = str(requirement.get("command", "") or "")
        if action or command:
            actions.append({"category": category, "action": action, "command": command})
    return actions


def _tcga_preflight_pipeline_command(args: argparse.Namespace, example_root: Path, experiment_path: Path) -> str:
    parts = ["paper-agent", "tcga-pipeline", "--example-root", str(example_root)]
    if getattr(args, "artifacts_dir", ""):
        parts.extend(["--artifacts-dir", str(_tcga_artifact_template_output_dir(args, example_root))])
    for attr, option in (
        ("main_csv", "--main-csv"),
        ("ablation_csv", "--ablation-csv"),
        ("sensitivity_csv", "--sensitivity-csv"),
        ("stats_csv", "--stats-csv"),
    ):
        value = str(getattr(args, attr, "") or "")
        if value:
            parts.extend([option, value])
    if experiment_path:
        parts.extend(["--experiment-results", str(experiment_path)])
    if getattr(args, "target_venue", ""):
        parts.extend(["--target-venue", str(args.target_venue)])
    for dataset in getattr(args, "dataset", []) or []:
        parts.extend(["--dataset", str(dataset)])
    if getattr(args, "online", False):
        parts.append("--online")
    if getattr(args, "offline", False):
        parts.append("--offline")
    if getattr(args, "disable_llm", False) and not getattr(args, "submission_grade", False):
        parts.append("--disable-llm")
    if getattr(args, "compile_latex", False):
        parts.append("--compile-latex")
    if getattr(args, "submission_grade", False):
        parts.append("--submission-grade")
    return " ".join(_powershell_arg(part) for part in parts)


def _add_preflight_check(
    checks: list[dict[str, object]],
    name: str,
    status: str,
    detail: str,
    blocking: list[str],
    *,
    blocking_item: bool = True,
) -> None:
    record = {"name": name, "status": status, "detail": detail}
    checks.append(record)
    if blocking_item and status == "FAIL":
        blocking.append(f"{name}: {detail}")


def _record_tcga_artifact_preflight(
    args: argparse.Namespace,
    example_root: Path,
    checks: list[dict[str, object]],
    blocking: list[str],
) -> bool:
    try:
        artifact_dir = _tcga_artifact_dir(args, example_root)
    except SystemExit as exc:
        _add_preflight_check(checks, "Artifact directory", "FAIL", str(exc), blocking)
        return False
    try:
        explicit_paths = {
            "main": _resolve_optional_file(args.main_csv, "Main result CSV"),
            "ablation": _resolve_optional_file(args.ablation_csv, "Ablation CSV"),
            "sensitivity": _resolve_optional_file(args.sensitivity_csv, "Sensitivity CSV"),
            "stats": _resolve_optional_file(args.stats_csv, "Statistical-test CSV"),
        }
    except SystemExit as exc:
        _add_preflight_check(checks, "Artifact CSV paths", "FAIL", str(exc), blocking)
        return False

    detected = (
        _detect_tcga_artifact_csvs(
            artifact_dir,
            method=args.method,
            baseline=args.baseline,
            metric=args.metric,
        )
        if artifact_dir
        else {}
    )
    has_input = bool(artifact_dir or any(explicit_paths.values()))
    _add_preflight_check(
        checks,
        "Artifact directory",
        "PASS" if has_input else "WARN",
        str(artifact_dir or "not provided; pass --artifacts-dir or explicit CSV paths"),
        blocking,
        blocking_item=False,
    )
    if not has_input:
        return False

    role_paths = {
        role: explicit_paths.get(role) or detected.get(role)
        for role in ("main", "ablation", "sensitivity", "stats")
    }
    required_roles = {
        "main": True,
        "ablation": bool(args.require_ablation),
        "sensitivity": bool(args.require_sensitivity),
        "stats": bool(args.require_statistical_tests),
    }
    datasets = list(args.dataset or [])
    required_ok = True
    for role in ("main", "ablation", "sensitivity", "stats"):
        label = f"Artifact {role}"
        path = role_paths.get(role)
        required = required_roles[role]
        if not path:
            status = "FAIL" if required else "WARN"
            _add_preflight_check(
                checks,
                label,
                status,
                f"missing; expected {_tcga_expected_artifact_schema(role)}",
                blocking,
                blocking_item=required,
            )
            required_ok = required_ok and not required
            continue
        diagnostic = _tcga_artifact_role_diagnostic(role, path, args=args, datasets=datasets)
        detail = str(diagnostic["detail"])
        issues = "; ".join(str(issue) for issue in diagnostic["issues"])
        if issues:
            detail = f"{detail}; {issues}"
        ok = bool(diagnostic["ok"])
        _add_preflight_check(
            checks,
            label,
            "PASS" if ok else "FAIL",
            detail,
            blocking,
            blocking_item=required,
        )
        required_ok = required_ok and (ok or not required)
        if role == "main" and diagnostic["datasets"]:
            datasets = list(diagnostic["datasets"])
    return required_ok


def _tcga_result_path(args: argparse.Namespace, example_root: Path) -> Path:
    if getattr(args, "experiment_results", ""):
        return _resolve_project_relative_path(args.experiment_results)
    return example_root / "results" / "tcga_results.md"


def _load_tcga_artifact_flow_summary(args: argparse.Namespace) -> tuple[Path | None, dict]:
    summary_value = str(getattr(args, "artifact_flow_summary", "") or "")
    if not summary_value:
        return None, {}
    summary_path = _resolve_project_relative_path(summary_value)
    if not summary_path.is_file():
        raise SystemExit(f"TCGA artifact flow summary not found: {summary_path}")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"TCGA artifact flow summary is invalid JSON: {summary_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"TCGA artifact flow summary must be a JSON object: {summary_path}")
    status = str(payload.get("status", "unknown"))
    if status != "pass":
        raise SystemExit(
            f"TCGA artifact flow summary is not pass: {summary_path} status={status}. "
            "Rerun tcga-demo-artifact-flow or validate the result artifacts before drafting."
        )
    if not str(payload.get("experiment_results", "") or ""):
        raise SystemExit(f"TCGA artifact flow summary is missing experiment_results: {summary_path}")
    return summary_path, payload


def _tcga_result_path_from_artifact_flow_summary(summary_path: Path, summary: dict) -> Path:
    result_value = str(summary.get("experiment_results", "") or "")
    result_path = Path(result_value)
    if result_path.is_absolute():
        return result_path
    summary_relative = summary_path.parent / result_path
    if summary_relative.exists():
        return summary_relative
    return _resolve_project_relative_path(result_value)


def _record_tcga_artifact_flow_summary(
    state: dict,
    summary_path: Path | None,
    summary: dict,
) -> None:
    if not summary_path or not summary:
        return
    artifacts = state.setdefault("artifacts", {})
    artifacts["tcga_artifact_flow_summary_path"] = str(summary_path)
    artifacts["tcga_artifact_flow_summary_status"] = str(summary.get("status", "unknown"))
    artifacts["tcga_artifact_flow_pipeline_phase"] = str(summary.get("pipeline_phase", "unknown"))
    artifacts["tcga_artifact_flow_artifacts_dir"] = str(summary.get("artifacts_dir", ""))
    artifact_files = summary.get("artifact_files", {})
    artifacts["tcga_artifact_flow_artifact_files"] = artifact_files if isinstance(artifact_files, dict) else {}
    artifacts["tcga_artifact_flow_validation"] = {
        "experiment_contract_status": summary.get("experiment_contract_status", "unknown"),
        "experiment_quality_status": summary.get("experiment_quality_status", "unknown"),
        "experiment_provenance_status": summary.get("experiment_provenance_status", "unknown"),
        "experiment_artifact_consistency_status": summary.get(
            "experiment_artifact_consistency_status",
            "unknown",
        ),
        "artifact_consistency_matched": summary.get("artifact_consistency_matched", 0),
        "artifact_consistency_missing": summary.get("artifact_consistency_missing", 0),
        "artifact_consistency_mismatched": summary.get("artifact_consistency_mismatched", 0),
    }


def _tcga_artifact_template_output_dir(args: argparse.Namespace, example_root: Path) -> Path:
    artifacts_dir = getattr(args, "artifacts_dir", "")
    if artifacts_dir:
        return _resolve_project_relative_path(artifacts_dir)
    return example_root / "results" / "logs"


def _tcga_artifact_template_command(args: argparse.Namespace, example_root: Path) -> str:
    style = getattr(args, "artifact_template_style", "long") or "long"
    parts = [
        "paper-agent",
        "tcga-artifact-template",
        "--output-dir",
        str(_tcga_artifact_template_output_dir(args, example_root)),
        "--style",
        style,
    ]
    for dataset in getattr(args, "dataset", []) or []:
        parts.extend(["--dataset", str(dataset)])
    return " ".join(_powershell_arg(part) for part in parts)


def _tcga_results_from_artifacts_command(args: argparse.Namespace, example_root: Path) -> str:
    parts = [
        "paper-agent",
        "tcga-results-from-artifacts",
        "--example-root",
        str(example_root),
        "--artifacts-dir",
        str(_tcga_artifact_template_output_dir(args, example_root)),
        "--output",
        str(_tcga_result_path(args, example_root)),
        "--strict",
    ]
    method = getattr(args, "method", "") or getattr(args, "expected_method", "")
    baseline = getattr(args, "baseline", "") or getattr(args, "expected_baseline", "")
    metric_values = list(getattr(args, "expected_metric", []) or [])
    metric = getattr(args, "metric", "") or (metric_values[0] if metric_values else "")
    for dataset in getattr(args, "dataset", []) or []:
        parts.extend(["--dataset", str(dataset)])
    for dataset in getattr(args, "expected_dataset", []) or []:
        parts.extend(["--dataset", str(dataset)])
    if method:
        parts.extend(["--method", str(method)])
    if baseline:
        parts.extend(["--baseline", str(baseline)])
    if metric:
        parts.extend(["--metric", str(metric)])
    return " ".join(_powershell_arg(part) for part in parts)


def _powershell_arg(value: object) -> str:
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:\\-]+", text):
        return text
    return "'" + text.replace("'", "''") + "'"


def _resolve_required_file(path_value: str, label: str) -> Path:
    path = _resolve_project_relative_path(path_value)
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def _resolve_optional_file(path_value: str, label: str) -> Path | None:
    if not path_value:
        return None
    return _resolve_required_file(path_value, label)


def _resolve_required_or_detected_file(
    explicit_path: str,
    detected_path: Path | None,
    label: str,
) -> Path:
    if explicit_path:
        return _resolve_required_file(explicit_path, label)
    if detected_path and detected_path.is_file():
        return detected_path
    raise SystemExit(
        f"{label} not found. Pass --main-csv explicitly or provide --artifacts-dir "
        "with a CSV containing the baseline and proposed-method result rows."
    )


def _resolve_optional_or_detected_file(
    explicit_path: str,
    detected_path: Path | None,
    label: str,
) -> Path | None:
    if explicit_path:
        return _resolve_required_file(explicit_path, label)
    return detected_path if detected_path and detected_path.is_file() else None


def _tcga_artifact_dir(args: argparse.Namespace, example_root: Path) -> Path | None:
    if args.artifacts_dir:
        path = _resolve_project_relative_path(args.artifacts_dir)
        if not path.is_dir():
            raise SystemExit(f"Artifact directory not found: {path}")
        return path
    default_path = example_root / "results" / "logs"
    return default_path if default_path.is_dir() else None


def _detect_tcga_artifact_csvs(
    artifact_dir: Path,
    *,
    method: str,
    baseline: str,
    metric: str,
) -> dict[str, Path]:
    scored: dict[str, list[tuple[int, str, Path]]] = {
        "main": [],
        "ablation": [],
        "sensitivity": [],
        "stats": [],
    }
    for csv_path in sorted(artifact_dir.rglob("*.csv")):
        try:
            rows = _read_csv_dicts(csv_path)
        except (csv.Error, OSError, UnicodeDecodeError):
            continue
        scores = _tcga_artifact_role_scores(
            csv_path,
            rows,
            method=method,
            baseline=baseline,
            metric=metric,
        )
        for role, score in scores.items():
            if score > 0:
                scored[role].append((score, str(csv_path).lower(), csv_path))
    detected = {}
    for role, candidates in scored.items():
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], item[1]))
        detected[role] = candidates[0][2]
    return detected


def _tcga_artifact_role_scores(
    csv_path: Path,
    rows: list[dict[str, str]],
    *,
    method: str,
    baseline: str,
    metric: str,
) -> dict[str, int]:
    headers = list(rows[0].keys()) if rows else []
    joined_rows = "\n".join(" ".join(row.values()) for row in rows[:50])
    name_text = f"{csv_path.parent.name} {csv_path.stem}".lower()
    source_text = f"{name_text}\n{' '.join(headers)}\n{joined_rows}"
    method_header = _csv_header(headers, "method", "model", "variant", "approach")
    dataset_header = _csv_header(headers, "dataset", "cohort", "cancer", "project")
    metric_header = _csv_header(headers, "metric", "measure")
    value_header = _csv_header(headers, "value", "score", "result", "mean", "estimate")
    comparison_header = _csv_header(headers, "comparison", "contrast", "pair")
    p_value_header = _csv_p_value_header(headers)
    parameter_header = _csv_header(headers, "parameter", "hyperparameter", "param")
    parameter_value_header = _csv_header(headers, "parameter_value", "param_value", "setting", "tested_value")

    scores = {"main": 0, "ablation": 0, "sensitivity": 0, "stats": 0}
    if re.search(r"\b(main|overall|performance|result|metric|eval|evaluation)\b", name_text):
        scores["main"] += 3
    if re.search(r"\b(ablation|ablat|variant|component)\b", name_text):
        scores["ablation"] += 6
    if re.search(r"\b(sensitivity|lambda|hyper|param|sweep)\b", name_text):
        scores["sensitivity"] += 6
    if re.search(r"\b(stat|stats|statistical|significance|pvalue|p-value|wilcoxon)\b", name_text):
        scores["stats"] += 6

    if method_header and _csv_rows_contain_label(rows, baseline) and _csv_rows_contain_label(rows, method):
        scores["main"] += 8
    if method_header and _has_dataset_metric_values(headers, dataset_header, metric_header, value_header, metric):
        scores["main"] += 3
    if method_header and _has_wide_dataset_metric(headers, metric):
        scores["main"] += 4

    if method_header and _rows_look_like_ablation(rows):
        scores["ablation"] += 7
    if _norm_key(method_header) == "variant":
        scores["ablation"] += 4
    if _has_wide_average_metric(headers, metric) and method_header:
        scores["ablation"] += 2

    if (parameter_header and parameter_value_header) or _has_parameter_like_header(headers):
        scores["sensitivity"] += 8
    if _has_wide_average_metric(headers, metric) and _has_parameter_like_header(headers):
        scores["sensitivity"] += 3

    if comparison_header and p_value_header:
        scores["stats"] += 10
    if re.search(r"\bp\s*[-_ ]?value\b|\bp\s*[<=>]\s*0?\.\d+", source_text, flags=re.I):
        scores["stats"] += 3
    return scores


def _csv_header(headers: list[str], *names: str) -> str:
    normalized = {_norm_key(header): header for header in headers}
    for name in names:
        key = _norm_key(name)
        if key in normalized:
            return normalized[key]
    for name in names:
        key = _norm_key(name)
        if key == "p":
            continue
        for header in headers:
            if key and key in _norm_key(header):
                return header
    return ""


def _csv_p_value_header(headers: list[str]) -> str:
    for header in headers:
        key = _norm_key(header)
        if key in {"p", "pvalue", "pval"}:
            return header
    return _csv_header(headers, "p_value", "p-value", "pval")


def _csv_rows_contain_label(rows: list[dict[str, str]], label: str) -> bool:
    label_norm = _norm_key(label)
    for row in rows:
        for value in row.values():
            value_norm = _norm_key(value)
            if label_norm and value_norm and (label_norm in value_norm or value_norm in label_norm):
                return True
    return False


def _has_dataset_metric_values(
    headers: list[str],
    dataset_header: str,
    metric_header: str,
    value_header: str,
    metric: str,
) -> bool:
    if dataset_header and value_header:
        return True
    if metric_header and value_header and _norm_key(metric) in _norm_key(metric_header + " " + value_header):
        return True
    return _has_wide_dataset_metric(headers, metric)


def _has_wide_dataset_metric(headers: list[str], metric: str) -> bool:
    metric_norm = _norm_key(metric)
    return any(metric_norm and metric_norm in _norm_key(header) and "average" not in header.lower() for header in headers)


def _has_wide_average_metric(headers: list[str], metric: str) -> bool:
    metric_norm = _norm_key(metric)
    return any(metric_norm and metric_norm in _norm_key(header) and re.search(r"\b(avg|average|mean)\b", header, flags=re.I) for header in headers)


def _has_parameter_like_header(headers: list[str]) -> bool:
    return any(
        re.search(r"\b(lambda(?:[_-]?[a-z]+)?|alpha|beta|gamma|parameter|hyper[-_ ]?parameter|dropout|weight|k)\b", header, flags=re.I)
        for header in headers
    )


def _rows_look_like_ablation(rows: list[dict[str, str]]) -> bool:
    text = "\n".join(" ".join(row.values()) for row in rows[:50])
    return bool(re.search(r"\b(w/o|without|no|ablat(?:e|ed|ion)|remove(?:d)?|minus)\b", text, flags=re.I))


def _print_detected_tcga_artifacts(detected: dict[str, Path]) -> None:
    print("Auto-detected artifacts:")
    for role in ("main", "ablation", "sensitivity", "stats"):
        if role in detected:
            print(f"- {role}: {detected[role]}")


def _tcga_artifact_role_diagnostic(
    role: str,
    path: Path,
    *,
    args: argparse.Namespace,
    datasets: list[str],
) -> dict[str, object]:
    try:
        rows = _read_csv_dicts(path)
    except (csv.Error, OSError, UnicodeDecodeError) as exc:
        return {
            "ok": False,
            "detail": f"{path}; unreadable CSV",
            "issues": [f"CSV read failed: {exc}"],
            "datasets": [],
        }
    headers = list(rows[0].keys()) if rows else []
    issues: list[str] = []
    parsed_values = 0
    parsed_datasets = list(datasets)
    try:
        if not rows:
            raise SystemExit("CSV has headers but no data rows.")
        if role == "main":
            parsed_datasets = datasets or _detect_result_datasets(rows, metric=args.metric)
            if not parsed_datasets:
                raise SystemExit("No datasets detected. Use --dataset or wide columns such as BLCA C-index.")
            parsed_values = sum(
                len(values)
                for _, values in _tcga_main_result_table(
                    rows,
                    datasets=parsed_datasets,
                    metric=args.metric,
                    method=args.method,
                    baseline=args.baseline,
                )
            )
        elif role == "ablation":
            parsed_values = len(
                _tcga_single_value_table(
                    rows,
                    label_column_candidates=("variant", "method"),
                    value_column_hint=args.metric,
                    table_label="Variant",
                    dataset_filter="Average",
                    allowed_label_patterns=(args.method, r"\b(w/o|without|no|ablat(?:e|ed|ion)|remove(?:d)?|minus)\b"),
                )
            )
        elif role == "sensitivity":
            parsed_values = len(
                _tcga_single_value_table(
                    rows,
                    label_column_candidates=("lambda_rec", "parameter_value", "value_name"),
                    value_column_hint=args.metric,
                    table_label="lambda_rec",
                    dataset_filter="Average",
                )
            )
        elif role == "stats":
            parsed_values = len(_tcga_stats_table(rows))
        else:
            raise SystemExit(f"Unknown TCGA artifact role: {role}")
    except SystemExit as exc:
        issues.append(str(exc))
        issues.append(f"Expected schema: {_tcga_expected_artifact_schema(role)}")
    column_preview = ", ".join(headers[:10]) if headers else "none"
    if len(headers) > 10:
        column_preview += ", ..."
    ok = not issues and parsed_values > 0
    if not issues and parsed_values <= 0:
        issues.append(f"No parseable values found. Expected schema: {_tcga_expected_artifact_schema(role)}")
        ok = False
    return {
        "ok": ok,
        "detail": f"{path}; rows={len(rows)}; columns={column_preview}; parsed_values={parsed_values}",
        "issues": issues,
        "datasets": parsed_datasets,
    }


def _tcga_artifact_role_label(role: str) -> str:
    return {
        "main": "Main result CSV",
        "ablation": "Ablation CSV",
        "sensitivity": "Sensitivity CSV",
        "stats": "Statistical-test CSV",
    }.get(role, role)


def _tcga_expected_artifact_schema(role: str) -> str:
    schemas = {
        "main": "wide `method,BLCA C-index,...` or long `method,dataset,metric,value` with baseline and proposed-method rows",
        "ablation": "`variant,Average C-index` or long `method,dataset=Average,metric,value` with full and w/o rows",
        "sensitivity": "`lambda_rec,Average C-index` or long `parameter,parameter_value,dataset=Average,metric,value` rows",
        "stats": "`comparison,metric,test,p_value` rows",
    }
    return schemas.get(role, "supported TCGA result CSV schema")


def _tcga_artifact_template_files(
    *,
    style: str,
    datasets: list[str],
    method: str,
    baseline: str,
    metric: str,
    seed: str,
) -> dict[str, str]:
    if style == "wide":
        return {
            "tcga_main_results.csv": _csv_text(
                [
                    ["method", *[f"{dataset} {metric}" for dataset in datasets]],
                    [baseline, *(["TODO"] * len(datasets))],
                    [method, *(["TODO"] * len(datasets))],
                ]
            ),
            "tcga_ablation.csv": _csv_text(
                [
                    ["variant", f"Average {metric}"],
                    [method, "TODO"],
                    ["w/o reconstruction loss", "TODO"],
                ]
            ),
            "tcga_sensitivity.csv": _csv_text(
                [
                    ["lambda_rec", f"Average {metric}"],
                    ["0.5", "TODO"],
                    ["1.0", "TODO"],
                ]
            ),
            "tcga_stats.csv": _csv_text(
                [
                    ["comparison", "metric", "test", "p_value"],
                    [f"{method} vs {baseline}", metric, "Wilcoxon signed-rank", "TODO"],
                ]
            ),
        }
    return {
        "tcga_main_results.csv": _csv_text(
            [
                ["method", "dataset", "metric", "fold", "seed", "value"],
                *[[baseline, dataset, metric, "0", seed, "TODO"] for dataset in datasets],
                *[[method, dataset, metric, "0", seed, "TODO"] for dataset in datasets],
            ]
        ),
        "tcga_ablation.csv": _csv_text(
            [
                ["method", "dataset", "metric", "fold", "seed", "value"],
                [method, "Average", metric, "0", seed, "TODO"],
                ["w/o reconstruction loss", "Average", metric, "0", seed, "TODO"],
            ]
        ),
        "tcga_sensitivity.csv": _csv_text(
            [
                ["parameter", "parameter_value", "dataset", "metric", "fold", "seed", "value"],
                ["lambda_rec", "0.5", "Average", metric, "0", seed, "TODO"],
                ["lambda_rec", "1.0", "Average", metric, "0", seed, "TODO"],
            ]
        ),
        "tcga_stats.csv": _csv_text(
            [
                ["comparison", "metric", "test", "p_value"],
                [f"{method} vs {baseline}", metric, "Wilcoxon signed-rank", "TODO"],
            ]
        ),
    }


def _tcga_artifact_template_bundle(
    *,
    style: str,
    datasets: list[str],
    method: str,
    baseline: str,
    metric: str,
    seed: str,
) -> dict[str, str]:
    files = _tcga_artifact_template_files(
        style=style,
        datasets=datasets,
        method=method,
        baseline=baseline,
        metric=metric,
        seed=seed,
    )
    files["EXPORT_CONTRACT.md"] = _tcga_artifact_export_contract(
        style=style,
        datasets=datasets,
        method=method,
        baseline=baseline,
        metric=metric,
    )
    files["ARTIFACT_SCHEMA.json"] = (
        json.dumps(
            _tcga_artifact_schema_manifest(
                style=style,
                datasets=datasets,
                method=method,
                baseline=baseline,
                metric=metric,
                seed=seed,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return files


def _tcga_artifact_export_contract(
    *,
    style: str,
    datasets: list[str],
    method: str,
    baseline: str,
    metric: str,
) -> str:
    dataset_text = ", ".join(datasets)
    return "\n".join(
        [
            "# TCGA Artifact Export Contract",
            "",
            "These files define the CSV artifacts that paper-agent can turn into a paper-facing result file.",
            "Replace every `TODO` with real trained-model outputs before using them for a draft.",
            "",
            f"- Style: {style}",
            f"- Proposed method label: `{method}`",
            f"- Baseline label: `{baseline}`",
            f"- Metric: `{metric}`",
            f"- Datasets: {dataset_text}",
            "",
            "## Required Files",
            "",
            "- `tcga_main_results.csv`: baseline and proposed-method performance per TCGA cohort.",
            "- `tcga_ablation.csv`: full method and removed-component variants on the average metric.",
            "- `tcga_sensitivity.csv`: hyperparameter sweep rows on the average metric.",
            "- `tcga_stats.csv`: statistical comparison rows with numeric `p_value`.",
            "- `ARTIFACT_SCHEMA.json`: machine-readable schema manifest for training-code exporters.",
            "",
            "## Validation Flow",
            "",
            "```powershell",
            "paper-agent tcga-artifacts-doctor --artifacts-dir . --summary artifact-doctor.json",
            "paper-agent tcga-results-from-artifacts --artifacts-dir . --strict",
            "paper-agent tcga-preflight --artifacts-dir . --submission-grade",
            "```",
            "",
            "Do not use synthetic, mock, or cohort-metadata-only values for submission claims.",
            "",
        ]
    )


def _tcga_artifact_schema_manifest(
    *,
    style: str,
    datasets: list[str],
    method: str,
    baseline: str,
    metric: str,
    seed: str,
) -> dict[str, object]:
    if style == "wide":
        main_columns = ["method", *[f"{dataset} {metric}" for dataset in datasets]]
        ablation_columns = ["variant", f"Average {metric}"]
        sensitivity_columns = ["lambda_rec", f"Average {metric}"]
    else:
        main_columns = ["method", "dataset", "metric", "fold", "seed", "value"]
        ablation_columns = ["method", "dataset", "metric", "fold", "seed", "value"]
        sensitivity_columns = ["parameter", "parameter_value", "dataset", "metric", "fold", "seed", "value"]
    roles = {
        "main": {
            "file": "tcga_main_results.csv",
            "required": True,
            "columns": main_columns,
            "accepted_schema": _tcga_expected_artifact_schema("main"),
            "requirements": [
                f"include one row or value for baseline label `{baseline}`",
                f"include one row or value for proposed method label `{method}`",
                f"cover every dataset: {', '.join(datasets)}",
                f"use metric `{metric}`",
            ],
        },
        "ablation": {
            "file": "tcga_ablation.csv",
            "required": True,
            "columns": ablation_columns,
            "accepted_schema": _tcga_expected_artifact_schema("ablation"),
            "requirements": [
                f"include full method label `{method}`",
                "include at least one removed-component variant such as `w/o reconstruction loss`",
                f"use average `{metric}` values",
            ],
        },
        "sensitivity": {
            "file": "tcga_sensitivity.csv",
            "required": True,
            "columns": sensitivity_columns,
            "accepted_schema": _tcga_expected_artifact_schema("sensitivity"),
            "requirements": [
                "include parameter names and tested values",
                f"use average `{metric}` values",
            ],
        },
        "stats": {
            "file": "tcga_stats.csv",
            "required": True,
            "columns": ["comparison", "metric", "test", "p_value"],
            "accepted_schema": _tcga_expected_artifact_schema("stats"),
            "requirements": [
                f"compare `{method}` against `{baseline}`",
                "use numeric p_value values",
            ],
        },
    }
    return {
        "schema_version": 1,
        "style": style,
        "method": method,
        "baseline": baseline,
        "metric": metric,
        "seed": seed,
        "datasets": datasets,
        "roles": roles,
        "validation_commands": [
            "paper-agent tcga-artifacts-doctor --artifacts-dir . --summary artifact-doctor.json",
            "paper-agent tcga-results-from-artifacts --artifacts-dir . --strict",
            "paper-agent tcga-preflight --artifacts-dir . --submission-grade",
        ],
        "notes": [
            "Replace every TODO with real trained-model outputs before drafting.",
            "Do not use synthetic, mock, or cohort-metadata-only values for submission claims.",
            "Keep method, baseline, metric, and dataset labels stable so validation can match paper tables to CSV rows.",
        ],
    }


def _csv_text(rows: list[list[str]]) -> str:
    return "\n".join(",".join(_csv_cell(cell) for cell in row) for row in rows) + "\n"


def _csv_cell(value: object) -> str:
    text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _tcga_default_datasets() -> list[str]:
    defaults = tcga_experiment_quality_kwargs().get("expected_datasets", [])
    return [str(item) for item in defaults] or ["BLCA", "BRCA", "LGG", "LUAD", "UCEC"]


def _tcga_provenance_sources(
    *,
    main_csv: Path,
    ablation_csv: Path | None,
    sensitivity_csv: Path | None,
    stats_csv: Path | None,
) -> list[tuple[str, Path, str]]:
    raw_sources = [
        ("Main result CSV", main_csv, "source values for main result table"),
        ("Ablation CSV", ablation_csv, "source values for ablation table"),
        ("Sensitivity CSV", sensitivity_csv, "source values for sensitivity table"),
        ("Statistical CSV", stats_csv, "source values for statistical tests"),
    ]
    grouped: dict[Path, tuple[list[str], list[str], Path]] = {}
    for name, path, description in raw_sources:
        if not path:
            continue
        key = path.resolve()
        if key not in grouped:
            grouped[key] = ([], [], path)
        grouped[key][0].append(name)
        grouped[key][1].append(description)
    sources = []
    for names, descriptions, path in grouped.values():
        if len(names) == 1:
            sources.append((names[0], path, descriptions[0]))
        else:
            roles = ", ".join(description.replace("source values for ", "") for description in descriptions)
            sources.append(("Combined result CSV", path, f"source values for {roles}"))
    return sources


def _read_csv_dicts(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {str(key or "").strip(): str(value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _detect_result_datasets(rows: list[dict[str, str]], *, metric: str) -> list[str]:
    datasets = []
    for row in rows:
        dataset = _csv_get(row, "dataset")
        if dataset and dataset.lower() != "average" and dataset not in datasets:
            datasets.append(dataset)
    if datasets:
        return datasets
    metric_norm = _norm_key(metric)
    for header in rows[0].keys() if rows else []:
        header_norm = _norm_key(header)
        if metric_norm and metric_norm in header_norm:
            dataset = header[: header_norm.find(metric_norm)].strip(" _-/")
            dataset = dataset or header.replace(metric, "").strip(" _-/")
            if dataset and dataset not in datasets:
                datasets.append(dataset)
    return datasets


def _tcga_main_result_table(
    rows: list[dict[str, str]],
    *,
    datasets: list[str],
    metric: str,
    method: str,
    baseline: str,
) -> list[tuple[str, list[str]]]:
    table = []
    missing = []
    for label in (baseline, method):
        values = []
        for dataset in datasets:
            value = _lookup_result_value(rows, label=label, dataset=dataset, metric=metric)
            if value is None:
                missing.append(f"{label} / {dataset} {metric}")
                values.append("TODO")
            else:
                values.append(_format_result_value(value))
        table.append((label, values))
    if missing:
        raise SystemExit("Main result CSV is missing required values: " + "; ".join(missing) + ".")
    return table


def _tcga_single_value_table(
    rows: list[dict[str, str]],
    *,
    label_column_candidates: tuple[str, ...],
    value_column_hint: str,
    table_label: str,
    dataset_filter: str = "",
    allowed_label_patterns: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    values_by_label: dict[str, list[float]] = {}
    for row in rows:
        if not _row_matches_metric(row, value_column_hint):
            continue
        if not _row_matches_dataset_filter(row, dataset_filter):
            continue
        label = ""
        for candidate in label_column_candidates:
            label = _csv_get(row, candidate)
            if label:
                break
        if not label:
            continue
        if allowed_label_patterns and not _label_matches_any_pattern(label, allowed_label_patterns):
            continue
        value = _lookup_row_value(row, value_column_hint)
        if value is None:
            continue
        values_by_label.setdefault(label, []).append(value)
    table = [
        (label, _format_result_value(statistics.mean(values)))
        for label, values in values_by_label.items()
        if values
    ]
    if not table and rows:
        raise SystemExit(f"Could not parse {table_label} values from CSV.")
    return table


def _tcga_stats_table(rows: list[dict[str, str]]) -> list[tuple[str, str, str, str]]:
    table = []
    for row in rows:
        comparison = _csv_get(row, "comparison")
        metric = _csv_get(row, "metric") or "C-index"
        test = _csv_get(row, "test") or "statistical test"
        p_value = _csv_get(row, "p_value") or _csv_get(row, "p-value") or _csv_get(row, "p")
        if comparison and p_value:
            table.append((comparison, metric, test, p_value))
    if rows and not table:
        raise SystemExit("Could not parse statistical-test values from CSV.")
    return table


def _lookup_result_value(
    rows: list[dict[str, str]],
    *,
    label: str,
    dataset: str,
    metric: str,
) -> float | None:
    label_norm = _norm_key(label)
    dataset_norm = _norm_key(dataset)
    metric_norm = _norm_key(metric)
    values = []
    for row in rows:
        method = _csv_get(row, "method") or _csv_get(row, "variant")
        if method and label_norm not in _norm_key(method) and _norm_key(method) not in label_norm:
            continue
        row_dataset = _csv_get(row, "dataset")
        row_metric = _csv_get(row, "metric")
        if row_dataset and _norm_key(row_dataset) != dataset_norm:
            continue
        if row_metric and metric_norm not in _norm_key(row_metric):
            continue
        value = _lookup_row_value(row, f"{dataset} {metric}")
        if value is not None:
            values.append(value)
            continue
        value = _lookup_row_value(row, metric)
        if value is not None and row_dataset:
            values.append(value)
    return statistics.mean(values) if values else None


def _lookup_row_value(row: dict[str, str], hint: str) -> float | None:
    hint_norm = _norm_key(hint)
    for key, value in row.items():
        key_norm = _norm_key(key)
        if key_norm in {"value", "score"} or (hint_norm and hint_norm in key_norm):
            parsed = _try_float(value)
            if parsed is not None:
                return parsed
    return None


def _row_matches_metric(row: dict[str, str], metric: str) -> bool:
    row_metric = _csv_get(row, "metric")
    return not row_metric or _norm_key(metric) in _norm_key(row_metric)


def _row_matches_dataset_filter(row: dict[str, str], dataset_filter: str) -> bool:
    if not dataset_filter:
        return True
    row_dataset = _csv_get(row, "dataset")
    return not row_dataset or _norm_key(row_dataset) == _norm_key(dataset_filter)


def _label_matches_any_pattern(label: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if not pattern:
            continue
        if _norm_key(pattern) and _norm_key(pattern) in _norm_key(label):
            return True
        try:
            if re.search(pattern, label, flags=re.I):
                return True
        except re.error:
            continue
    return False


def _csv_get(row: dict[str, str], name: str) -> str:
    target = _norm_key(name)
    for key, value in row.items():
        if _norm_key(key) == target:
            return value.strip()
    return ""


def _norm_key(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _try_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _format_result_value(value: float | None) -> str:
    if value is None:
        return "TODO"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _render_tcga_results_markdown(
    *,
    metric: str,
    datasets: list[str],
    main_table: list[tuple[str, list[str]]],
    ablation_table: list[tuple[str, str]],
    sensitivity_table: list[tuple[str, str]],
    stats_table: list[tuple[str, str, str, str]],
    provenance_sources: list[tuple[str, Path, str]],
    output_path: Path,
) -> str:
    lines = [
        "# Real Experiment Results",
        "",
        "Generated from local result artifacts. Verify every value before submission.",
        "",
        "## Main Results",
        "",
        f"Metric: {metric}. Higher is better.",
        "",
        "| Method | " + " | ".join(f"{dataset} {metric}" for dataset in datasets) + " |",
        "|---|" + "|".join("---:" for _ in datasets) + "|",
    ]
    for label, values in main_table:
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    if ablation_table:
        lines.extend(["", "## Ablation Study", "", f"Metric: Average {metric}. Higher is better.", "", f"| Variant | Average {metric} |", "|---|---:|"])
        lines.extend(f"| {label} | {value} |" for label, value in ablation_table)
    if sensitivity_table:
        lines.extend(["", "## Sensitivity Analysis", "", f"Metric: Average {metric}. Higher is better.", "", f"| lambda_rec | Average {metric} |", "|---:|---:|"])
        lines.extend(f"| {label} | {value} |" for label, value in sensitivity_table)
    if stats_table:
        lines.extend(["", "## Statistical Testing", "", "| Comparison | Metric | Test | p-value |", "|---|---|---|---:|"])
        lines.extend(f"| {comparison} | {metric_name} | {test} | {p_value} |" for comparison, metric_name, test, p_value in stats_table)
    lines.extend(["", "## Result Provenance", "", "| Artifact | Path | SHA256 | Description |", "|---|---|---|---|"])
    for name, path, description in provenance_sources:
        rel_path = _relative_artifact_path(path, output_path.parent)
        lines.append(f"| {name} | {rel_path} | {_sha256_file(path)} | {description}; seed=see-artifact; fold=see-artifact |")
    return "\n".join(lines) + "\n"


def _relative_artifact_path(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path, root).replace("\\", "/")
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print_doctor_check(name: str, ok: bool, detail: str) -> dict[str, object]:
    status = "PASS" if ok else "FAIL"
    print(f"- {name}: {status} ({detail})")
    return {"name": name, "status": status, "detail": detail}


def _run_tcga_pipeline(args: argparse.Namespace) -> None:
    if getattr(args, "submission_grade", False):
        _apply_submission_grade_defaults(args)
    example_root = Path(args.example_root)
    experiment_path = _tcga_result_path(args, example_root)
    args.experiment_results = str(experiment_path)
    result_generation_mode = "generated_from_artifacts"
    doctor_mode = "completed"

    if args.skip_result_generation:
        result_generation_mode = "skipped_existing"
        if not experiment_path.is_file():
            summary_path = _write_tcga_pipeline_status(
                args,
                example_root,
                phase="result_file_check",
                status="blocked",
                blocking_items=[f"TCGA result file is missing: {experiment_path}"],
                missing_inputs=[f"Complete paper-facing result Markdown: {experiment_path}"],
                next_action="Create a complete result file or rerun without --skip-result-generation.",
                next_command=_tcga_artifact_template_command(args, example_root),
            )
            print(f"Pipeline summary written to {summary_path}")
            raise SystemExit(
                f"Cannot skip result generation; TCGA result file is missing: {experiment_path}. "
                f"Pipeline summary: {summary_path}"
            )
        print(f"TCGA pipeline: using existing result file {experiment_path}")
    else:
        print(f"TCGA pipeline: generating result file {experiment_path}")
        result_args = argparse.Namespace(**vars(args))
        result_args.output = str(experiment_path)
        result_args.strict = True
        try:
            _run_tcga_results_from_artifacts(result_args)
        except SystemExit as exc:
            if getattr(args, "write_artifact_template", False):
                template_dir = _tcga_artifact_template_output_dir(args, example_root)
                template_args = argparse.Namespace(
                    output_dir=str(template_dir),
                    method=args.method,
                    baseline=args.baseline,
                    metric=args.metric,
                    seed="2026",
                    dataset=list(args.dataset or []),
                    style=args.artifact_template_style,
                    force=False,
                )
                print("TCGA pipeline: result artifacts are missing or incomplete; writing artifact templates")
                try:
                    _run_tcga_artifact_template(template_args)
                except SystemExit as template_exc:
                    summary_path = _write_tcga_pipeline_status(
                        args,
                        example_root,
                        phase="artifact_template_write",
                        status="blocked",
                        blocking_items=[
                            f"Could not generate result file: {exc}",
                            f"Could not write artifact templates: {template_exc}",
                        ],
                        missing_inputs=_tcga_pipeline_missing_artifact_inputs(template_dir),
                        next_action="Resolve the template write failure, then create real result CSV artifacts.",
                        next_command=_tcga_artifact_template_command(args, example_root),
                    )
                    print(f"Pipeline summary written to {summary_path}")
                    raise SystemExit(
                        "TCGA pipeline could not generate results and could not write artifact templates. "
                        f"Original result error: {exc}. Template error: {template_exc}. "
                        f"Pipeline summary: {summary_path}"
                    ) from template_exc
                summary_path = _write_tcga_pipeline_status(
                    args,
                    example_root,
                    phase="artifact_template_written",
                    status="blocked",
                    blocking_items=[f"Result CSV artifacts are missing or incomplete: {exc}"],
                    missing_inputs=_tcga_pipeline_missing_artifact_inputs(template_dir),
                    next_action="Replace every TODO in the generated CSV templates with real trained-model outputs.",
                    next_command=_tcga_pipeline_rerun_command(args, example_root),
                    outputs={
                        "artifact_template_dir": str(template_dir),
                        "artifact_contract_path": str(template_dir / "EXPORT_CONTRACT.md"),
                        "artifact_schema_path": str(template_dir / "ARTIFACT_SCHEMA.json"),
                    },
                )
                print(f"Pipeline summary written to {summary_path}")
                raise SystemExit(
                    "TCGA pipeline stopped after writing artifact templates. Replace every TODO with real "
                    f"trained-model outputs, then rerun tcga-pipeline. Pipeline summary: {summary_path}"
                ) from exc
            summary_path = _write_tcga_pipeline_status(
                args,
                example_root,
                phase="result_artifact_detection",
                status="blocked",
                blocking_items=[f"Result CSV artifacts are missing or incomplete: {exc}"],
                missing_inputs=_tcga_pipeline_missing_artifact_inputs(
                    _tcga_artifact_template_output_dir(args, example_root)
                ),
                next_action="Create real-result CSV artifacts before drafting.",
                next_command=_tcga_artifact_template_command(args, example_root),
            )
            print(f"Pipeline summary written to {summary_path}")
            raise SystemExit(_tcga_pipeline_artifact_failure_message(args, example_root, exc, summary_path)) from exc

    if args.skip_doctor:
        doctor_mode = "skipped"
        print("TCGA pipeline: skipping doctor checks")
    else:
        print("TCGA pipeline: running doctor checks")
        doctor_args = argparse.Namespace(**vars(args))
        doctor_args.write_template = False
        doctor_args.live_llm = bool(args.live_llm_doctor)
        try:
            _run_tcga_doctor(doctor_args)
        except SystemExit as exc:
            summary_path = _write_tcga_pipeline_status(
                args,
                example_root,
                phase="doctor_checks",
                status="blocked",
                blocking_items=[str(exc)],
                missing_inputs=_tcga_pipeline_doctor_missing_inputs(args, example_root),
                next_action="Fix the TCGA doctor blocking items, then rerun tcga-pipeline.",
                next_command=_tcga_doctor_command(args, example_root),
            )
            print(f"Pipeline summary written to {summary_path}")
            raise SystemExit(f"{exc} Pipeline summary: {summary_path}") from exc

    print("TCGA pipeline: drafting paper")
    try:
        _run_tcga_draft(args)
    except SystemExit as exc:
        plan = _tcga_pipeline_draft_failure_plan(args, example_root, str(exc))
        summary_path = _write_tcga_pipeline_status(
            args,
            example_root,
            phase=plan["phase"],
            status="blocked",
            blocking_items=[str(exc)],
            missing_inputs=plan["missing_inputs"],
            next_action=plan["next_action"],
            next_command=plan["next_command"],
            diagnostics=plan.get("diagnostics"),
        )
        print(f"Pipeline summary written to {summary_path}")
        raise SystemExit(f"{exc} Pipeline summary: {summary_path}") from exc
    summary_path = _write_tcga_pipeline_success_summary(
        args,
        example_root,
        result_generation_mode=result_generation_mode,
        doctor_mode=doctor_mode,
    )
    print(f"Pipeline summary written to {summary_path}")
    print("TCGA pipeline completed.")


def _run_tcga_readiness_schema(args: argparse.Namespace) -> None:
    payload = _tcga_readiness_contract_example() if args.example else _tcga_readiness_contract_schema()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output_path = _resolve_project_relative_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"TCGA readiness {'example' if args.example else 'schema'} written to {output_path}")
    else:
        print(text)


def _tcga_readiness_contract_schema() -> dict[str, object]:
    requirement_schema = {
        "type": "object",
        "required": ["status", "required", "detail", "next_action", "command"],
        "properties": {
            "status": {"type": "string", "enum": list(TCGA_READINESS_REQUIREMENT_STATUSES)},
            "required": {"type": "boolean"},
            "detail": {"type": "string"},
            "next_action": {"type": "string"},
            "command": {"type": "string"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://paper-agent.local/schemas/tcga-readiness-contract/v1.json",
        "title": "TCGA readiness_contract",
        "schema_version": TCGA_READINESS_CONTRACT_SCHEMA_VERSION,
        "type": "object",
        "required": [
            "schema_version",
            "status",
            "submission_grade",
            "ready_for_submission_grade",
            "ready_for_deterministic_draft",
            "blocking_categories",
            "requirements",
            "next_actions",
        ],
        "properties": {
            "schema_version": {"const": TCGA_READINESS_CONTRACT_SCHEMA_VERSION},
            "status": {"type": "string", "enum": ["ready", "blocked"]},
            "submission_grade": {"type": "boolean"},
            "pipeline_phase": {"type": "string"},
            "ready_for_submission_grade": {"type": "boolean"},
            "ready_for_deterministic_draft": {"type": "boolean"},
            "blocking_categories": {
                "type": "array",
                "items": {"type": "string", "enum": list(TCGA_READINESS_CONTRACT_CATEGORIES)},
            },
            "requirements": {
                "type": "object",
                "properties": {
                    category: requirement_schema for category in TCGA_READINESS_CONTRACT_CATEGORIES
                },
                "additionalProperties": requirement_schema,
            },
            "next_actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["category", "action", "command"],
                    "properties": {
                        "category": {"type": "string"},
                        "action": {"type": "string"},
                        "command": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "artifact_warnings": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def _tcga_readiness_contract_example() -> dict[str, object]:
    return {
        "schema_version": TCGA_READINESS_CONTRACT_SCHEMA_VERSION,
        "status": "blocked",
        "submission_grade": True,
        "pipeline_phase": "llm_preflight",
        "ready_for_submission_grade": False,
        "ready_for_deterministic_draft": False,
        "blocking_categories": ["llm"],
        "requirements": {
            "venue_network": {
                "status": "pass",
                "required": True,
                "detail": "online",
                "next_action": "",
                "command": "",
            },
            "baseline_pdf": {
                "status": "pass",
                "required": True,
                "detail": "D:/code/agent/example/baseline/baseline.pdf",
                "next_action": "",
                "command": "",
            },
            "code_path": {
                "status": "pass",
                "required": True,
                "detail": "D:/code/agent/example/code/hyper-protosurv",
                "next_action": "",
                "command": "",
            },
            "result_artifacts": {
                "status": "pass",
                "required": False,
                "detail": "main=PASS; ablation=PASS; sensitivity=PASS; stats=PASS",
                "next_action": "",
                "command": "",
            },
            "experiment_results": {
                "status": "pass",
                "required": True,
                "detail": "D:/code/agent/example/results/tcga_results.md",
                "next_action": "",
                "command": "",
            },
            "llm": {
                "status": "fail",
                "required": True,
                "detail": "Provider quota or configuration blocked the live LLM preflight.",
                "next_action": "Fix LLM configuration or provider quota before LLM drafting.",
                "command": "paper-agent llm-doctor",
            },
            "latex": {
                "status": "pass",
                "required": True,
                "detail": "tectonic.exe",
                "next_action": "",
                "command": "",
            },
        },
        "next_actions": [
            {
                "category": "llm",
                "action": "Fix LLM configuration or provider quota before LLM drafting.",
                "command": "paper-agent llm-doctor",
            }
        ],
        "artifact_warnings": [],
    }


def _tcga_pipeline_artifact_failure_message(
    args: argparse.Namespace,
    example_root: Path,
    original_error: BaseException,
    summary_path: Path | None = None,
) -> str:
    command = _tcga_artifact_template_command(args, example_root)
    message = (
        "TCGA pipeline cannot generate the paper-facing result file because result CSV artifacts "
        f"are missing or incomplete. Original error: {original_error}. "
        f"Next step: run `{command}`, replace every TODO with real trained-model outputs, "
        "then rerun tcga-pipeline. Or pass --write-artifact-template to have the pipeline "
        "write these templates and stop automatically."
    )
    if summary_path:
        message += f" Pipeline summary: {summary_path}"
    return message


def _write_tcga_pipeline_status(
    args: argparse.Namespace,
    example_root: Path,
    *,
    phase: str,
    status: str,
    blocking_items: list[str],
    missing_inputs: list[str],
    next_action: str,
    next_command: str,
    outputs: dict[str, str] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> Path:
    output_dir = Path(args.output_dir)
    experiment_path = _tcga_result_path(args, example_root)
    artifact_dir = _tcga_artifact_template_output_dir(args, example_root)
    readiness_contract = _tcga_pipeline_failure_readiness_contract(
        args,
        example_root,
        phase=phase,
        blocking_items=blocking_items,
        missing_inputs=missing_inputs,
        next_action=next_action,
        next_command=next_command,
        diagnostics=diagnostics or {},
    )
    summary = {
        "project_name": args.project_name or _default_project_name(output_dir),
        "target_venue": args.target_venue,
        "status": status,
        "pipeline_phase": phase,
        "inputs": {
            "example_root": str(example_root),
            "experiment_results_path": str(experiment_path),
            "artifact_dir": str(artifact_dir),
            "target_venue": args.target_venue,
            "submission_grade": bool(args.submission_grade),
            "disable_llm": bool(args.disable_llm),
            "skip_result_generation": bool(args.skip_result_generation),
            "write_artifact_template": bool(getattr(args, "write_artifact_template", False)),
        },
        "blocking_items": blocking_items,
        "missing_inputs": missing_inputs,
        "readiness_contract": readiness_contract,
        "next_actions": readiness_contract["next_actions"],
        "next_action": next_action,
        "next_command": next_command,
        "diagnostics": diagnostics or {},
        "outputs": outputs or {},
    }
    return _write_run_summary_data(summary, output_dir / "RUN_SUMMARY.json")


def _tcga_pipeline_failure_readiness_contract(
    args: argparse.Namespace,
    example_root: Path,
    *,
    phase: str,
    blocking_items: list[str],
    missing_inputs: list[str],
    next_action: str,
    next_command: str,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    requirements: dict[str, dict[str, object]] = {}

    def add_requirement(
        category: str,
        status: str,
        detail: str,
        *,
        required: bool = True,
        action: str = "",
        command: str = "",
    ) -> None:
        requirements[category] = {
            "status": status,
            "required": required,
            "detail": detail,
            "next_action": action,
            "command": command,
        }

    submission_grade = bool(getattr(args, "submission_grade", False))
    offline = bool(getattr(args, "offline", False))
    add_requirement(
        "venue_network",
        "fail" if submission_grade and offline else "pass",
        "submission-grade requires online mode" if submission_grade and offline else "configured",
        required=submission_grade,
        action="Use --online or remove --offline for submission-grade runs." if submission_grade and offline else "",
    )

    try:
        baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
        baseline_status = "pass"
        baseline_detail = str(baseline_pdf)
        baseline_action = ""
    except SystemExit as exc:
        baseline_status = "fail"
        baseline_detail = str(exc)
        baseline_action = f"Place the baseline PDF under {example_root / 'baseline'}."
    add_requirement("baseline_pdf", baseline_status, baseline_detail, action=baseline_action)

    code_path = example_root / "code" / "hyper-protosurv"
    add_requirement(
        "code_path",
        "pass" if code_path.is_dir() else "fail",
        str(code_path),
        action=f"Place the project code under {code_path}." if not code_path.is_dir() else "",
    )

    artifact_dir = _tcga_artifact_template_output_dir(args, example_root)
    artifact_phase = phase in {
        "result_artifact_detection",
        "artifact_template_write",
        "artifact_template_written",
    }
    artifact_status = "fail" if artifact_phase and phase != "artifact_template_written" else "ready_to_fill" if phase == "artifact_template_written" else "pass"
    artifact_action = (
        "Fill the generated CSV templates with real trained-model outputs."
        if phase == "artifact_template_written"
        else "Create or repair real-result CSV artifacts before drafting."
        if artifact_phase
        else ""
    )
    artifact_command = _tcga_artifact_template_command(args, example_root) if artifact_phase else ""
    add_requirement(
        "result_artifacts",
        artifact_status,
        "; ".join(missing_inputs) if artifact_phase else str(artifact_dir),
        required=artifact_phase,
        action=artifact_action,
        command=artifact_command,
    )

    experiment_path = _tcga_result_path(args, example_root)
    if phase == "artifact_template_written":
        result_status = "ready_to_generate"
        result_action = "Rerun tcga-pipeline after replacing template TODO values."
        result_command = _tcga_pipeline_rerun_command(args, example_root)
    elif phase in {"result_file_check", "draft_result_validation", "result_artifact_detection", "artifact_template_write"}:
        result_status = "fail"
        result_action = "Create or repair the strict TCGA result Markdown before drafting."
        result_command = _validate_results_command(args, example_root) if phase == "draft_result_validation" else ""
    else:
        result_status = "pass"
        result_action = ""
        result_command = ""
    add_requirement(
        "experiment_results",
        result_status,
        str(experiment_path),
        action=result_action,
        command=result_command,
    )

    llm_config = load_llm_config()
    llm_status = "fail" if phase == "llm_preflight" else "disabled" if getattr(args, "disable_llm", False) and not submission_grade else "pass"
    llm_diagnostics = diagnostics.get("llm", {}) if isinstance(diagnostics, dict) else {}
    llm_detail = (
        str(llm_diagnostics.get("diagnosis", "LLM preflight failed."))
        if isinstance(llm_diagnostics, dict) and phase == "llm_preflight"
        else _llm_config_label(llm_config)
    )
    add_requirement(
        "llm",
        llm_status,
        llm_detail,
        required=bool(submission_grade or not getattr(args, "disable_llm", False)),
        action="Fix LLM configuration or provider quota before LLM drafting." if phase == "llm_preflight" else "",
        command="paper-agent llm-doctor" if phase == "llm_preflight" else "",
    )

    latex_status = _latex_toolchain_status()
    latex_failed = phase == "latex_validation"
    add_requirement(
        "latex",
        "fail" if latex_failed else "pass" if latex_status.get("available") else "warn",
        str(latex_status.get("preferred_tool") or latex_status.get("install_hint") or "not found"),
        required=bool(submission_grade or getattr(args, "compile_latex", False)),
        action="Install a local LaTeX compiler or rerun without submission-grade compilation." if latex_failed else "",
        command="paper-agent latex-doctor" if latex_failed else "",
    )

    if phase in {"doctor_checks", "draft_generation", "llm_generation"}:
        add_requirement(
            "pipeline_stage",
            "fail",
            "; ".join(blocking_items) or phase,
            action=next_action,
            command=next_command,
        )

    blocking_categories = [
        category for category, requirement in requirements.items() if requirement["status"] == "fail"
    ]
    next_actions = _tcga_preflight_next_actions(requirements)
    if not next_actions and (next_action or next_command):
        next_actions.append({"category": phase, "action": next_action, "command": next_command})
    return {
        "schema_version": TCGA_READINESS_CONTRACT_SCHEMA_VERSION,
        "status": "blocked",
        "submission_grade": submission_grade,
        "pipeline_phase": phase,
        "ready_for_submission_grade": False,
        "ready_for_deterministic_draft": False,
        "blocking_categories": blocking_categories,
        "requirements": requirements,
        "next_actions": next_actions,
    }


def _tcga_pipeline_success_readiness_contract(
    args: argparse.Namespace,
    example_root: Path,
    *,
    summary: dict[str, object],
    result_generation_mode: str,
    doctor_mode: str,
) -> dict[str, object]:
    submission_grade = bool(getattr(args, "submission_grade", False))
    compile_latex = bool(getattr(args, "compile_latex", False) or submission_grade)
    disable_llm = bool(getattr(args, "disable_llm", False))
    experiment_path = _tcga_result_path(args, example_root)
    artifact_dir = _tcga_artifact_template_output_dir(args, example_root)
    baseline_dir = example_root / "baseline"
    baseline_pdfs = sorted(baseline_dir.glob("*.pdf")) if baseline_dir.is_dir() else []
    code_path = example_root / "code" / "hyper-protosurv"
    latex_status = _latex_toolchain_status()
    package = summary.get("artifacts", {})
    package_status = ""
    if isinstance(package, dict):
        submission_package = package.get("submission_package", {})
        if isinstance(submission_package, dict):
            package_status = str(submission_package.get("status", "") or "")

    requirements = {
        "venue_network": {
            "status": "pass",
            "required": submission_grade,
            "detail": "online submission-grade run" if submission_grade else "not required",
            "next_action": "",
            "command": "",
        },
        "baseline_pdf": {
            "status": "pass",
            "required": True,
            "detail": str(baseline_pdfs[0] if baseline_pdfs else baseline_dir),
            "next_action": "",
            "command": "",
        },
        "code_path": {
            "status": "pass" if code_path.is_dir() else "warn",
            "required": True,
            "detail": str(code_path),
            "next_action": "" if code_path.is_dir() else "Restore the project code directory for reproducible reruns.",
            "command": "",
        },
        "result_artifacts": {
            "status": "pass",
            "required": result_generation_mode == "generated_from_artifacts",
            "detail": f"{artifact_dir}; result_generation={result_generation_mode}",
            "next_action": "",
            "command": "",
        },
        "experiment_results": {
            "status": "pass",
            "required": True,
            "detail": str(experiment_path),
            "next_action": "",
            "command": "",
        },
        "llm": {
            "status": "disabled" if disable_llm and not submission_grade else "pass",
            "required": bool(submission_grade or not disable_llm),
            "detail": _llm_config_label(load_llm_config()),
            "next_action": "",
            "command": "",
        },
        "latex": {
            "status": "pass" if compile_latex or latex_status.get("available") else "warn",
            "required": compile_latex,
            "detail": package_status
            or str(latex_status.get("preferred_tool") or latex_status.get("install_hint") or "not found"),
            "next_action": "",
            "command": "",
        },
        "pipeline_stage": {
            "status": "pass",
            "required": True,
            "detail": f"tcga_pipeline_complete; doctor_checks={doctor_mode}",
            "next_action": "",
            "command": "",
        },
    }
    next_actions = [
        {
            "category": "pipeline_stage",
            "action": "Review the generated draft package and validate the result contract before submission.",
            "command": _validate_results_command(args, example_root),
        }
    ]
    return {
        "schema_version": TCGA_READINESS_CONTRACT_SCHEMA_VERSION,
        "status": "ready",
        "submission_grade": submission_grade,
        "pipeline_phase": "tcga_pipeline_complete",
        "ready_for_submission_grade": submission_grade,
        "ready_for_deterministic_draft": True,
        "blocking_categories": [],
        "requirements": requirements,
        "next_actions": next_actions,
        "artifact_warnings": [],
    }


def _write_tcga_pipeline_success_summary(
    args: argparse.Namespace,
    example_root: Path,
    *,
    result_generation_mode: str,
    doctor_mode: str,
) -> Path:
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "RUN_SUMMARY.json"
    summary = _read_json_object(summary_path, "TCGA pipeline draft summary")
    experiment_path = _tcga_result_path(args, example_root)
    artifact_dir = _tcga_artifact_template_output_dir(args, example_root)
    outputs = summary.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
    outputs["pipeline_summary_path"] = str(summary_path)
    summary["outputs"] = outputs
    summary["status"] = "pass"
    summary["pipeline_phase"] = "tcga_pipeline_complete"
    summary["pipeline_status"] = "pass"
    summary["pipeline"] = {
        "status": "pass",
        "phase": "tcga_pipeline_complete",
        "example_root": str(example_root),
        "experiment_results_path": str(experiment_path),
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "result_generation": result_generation_mode,
        "doctor_checks": doctor_mode,
        "draft_summary_path": str(summary_path),
        "submission_grade": bool(getattr(args, "submission_grade", False)),
        "disable_llm": bool(getattr(args, "disable_llm", False)),
        "zip_path": str(getattr(args, "zip", "") or ""),
    }
    summary["blocking_items"] = []
    summary["missing_inputs"] = []
    summary["next_action"] = "Review the generated draft package and replace demo or weak evidence before submission."
    summary["next_command"] = _validate_results_command(args, example_root)
    readiness_contract = _tcga_pipeline_success_readiness_contract(
        args,
        example_root,
        summary=summary,
        result_generation_mode=result_generation_mode,
        doctor_mode=doctor_mode,
    )
    summary["readiness_contract"] = readiness_contract
    summary["next_actions"] = readiness_contract["next_actions"]
    return _write_run_summary_data(summary, summary_path)


def _tcga_pipeline_missing_artifact_inputs(artifact_dir: Path) -> list[str]:
    return [
        f"{artifact_dir / 'tcga_main_results.csv'} with baseline and proposed-method values for every TCGA cohort",
        f"{artifact_dir / 'tcga_ablation.csv'} with full-method and ablated-variant values",
        f"{artifact_dir / 'tcga_sensitivity.csv'} with hyperparameter sensitivity values",
        f"{artifact_dir / 'tcga_stats.csv'} with statistical-test p-values",
    ]


def _tcga_pipeline_rerun_command(args: argparse.Namespace, example_root: Path) -> str:
    parts = [
        "paper-agent",
        "tcga-pipeline",
        "--example-root",
        str(example_root),
        "--artifacts-dir",
        str(_tcga_artifact_template_output_dir(args, example_root)),
        "--output-dir",
        str(Path(args.output_dir)),
    ]
    if args.zip:
        parts.extend(["--zip", args.zip])
    if args.target_venue:
        parts.extend(["--target-venue", args.target_venue])
    for dataset in getattr(args, "dataset", []) or []:
        parts.extend(["--dataset", str(dataset)])
    if args.submission_grade:
        parts.append("--submission-grade")
    if args.disable_llm:
        parts.append("--disable-llm")
    return " ".join(_powershell_arg(part) for part in parts)


def _tcga_doctor_command(args: argparse.Namespace, example_root: Path) -> str:
    parts = [
        "paper-agent",
        "tcga-doctor",
        "--example-root",
        str(example_root),
    ]
    if args.experiment_results:
        parts.extend(["--experiment-results", args.experiment_results])
    if args.submission_grade:
        parts.append("--submission-grade")
    if args.live_llm_doctor:
        parts.append("--live-llm")
    return " ".join(_powershell_arg(part) for part in parts)


def _validate_results_command(args: argparse.Namespace, example_root: Path) -> str:
    parts = [
        "paper-agent",
        "validate-results",
        "--experiment-results",
        str(_tcga_result_path(args, example_root)),
        "--strict",
    ]
    for dataset in getattr(args, "dataset", []) or []:
        parts.extend(["--expected-dataset", str(dataset)])
    for dataset in getattr(args, "expected_dataset", []) or []:
        parts.extend(["--expected-dataset", str(dataset)])
    for metric in getattr(args, "expected_metric", []) or []:
        parts.extend(["--expected-metric", str(metric)])
    method = getattr(args, "method", "") or getattr(args, "expected_method", "")
    baseline = getattr(args, "baseline", "") or getattr(args, "expected_baseline", "")
    if method:
        parts.extend(["--expected-method", str(method)])
    if baseline:
        parts.extend(["--expected-baseline", str(baseline)])
    return " ".join(_powershell_arg(part) for part in parts)


def _tcga_pipeline_doctor_missing_inputs(args: argparse.Namespace, example_root: Path) -> list[str]:
    missing: list[str] = []
    try:
        _resolve_baseline_pdf(str(example_root / "baseline"))
    except SystemExit:
        missing.append(f"Baseline PDF under {example_root / 'baseline'}")
    code_path = example_root / "code" / "hyper-protosurv"
    if not code_path.is_dir():
        missing.append(f"Hyper-ProtoSurv code directory: {code_path}")
    experiment_path = _tcga_result_path(args, example_root)
    if not experiment_path.is_file():
        missing.append(f"Strict TCGA result file: {experiment_path}")
    if args.submission_grade and not load_llm_config().configured:
        missing.append("Configured LLM API key/model for submission-grade drafting")
    if args.submission_grade and not _latex_toolchain_status()["available"]:
        missing.append("Local LaTeX compiler for submission-grade validation")
    return missing or ["Review TCGA doctor output for the blocking check names."]


def _tcga_pipeline_draft_failure_plan(
    args: argparse.Namespace,
    example_root: Path,
    error_message: str,
) -> dict[str, object]:
    lower_message = error_message.lower()
    if "llm" in lower_message and ("preflight" in lower_message or "configured" in lower_message or "quota" in lower_message):
        llm_config = load_llm_config()
        return {
            "phase": "llm_preflight",
            "missing_inputs": ["Configured LLM API key/model with available provider quota"],
            "next_action": "Fix LLM configuration or provider quota, then rerun tcga-pipeline.",
            "next_command": "paper-agent llm-doctor",
            "diagnostics": {"llm": _llm_failure_diagnostics(llm_config, error_message)},
        }
    if "latex" in lower_message or "compiler" in lower_message:
        return {
            "phase": "latex_validation",
            "missing_inputs": ["Local LaTeX compiler or valid LaTeX project assets"],
            "next_action": "Fix the local LaTeX toolchain or rerun without submission-grade compilation.",
            "next_command": "paper-agent latex-doctor",
        }
    if "experiment result" in lower_message or "result validation" in lower_message:
        return {
            "phase": "draft_result_validation",
            "missing_inputs": [f"Strict-acceptable TCGA result file: {_tcga_result_path(args, example_root)}"],
            "next_action": "Validate and repair the result file before drafting.",
            "next_command": _validate_results_command(args, example_root),
        }
    if "llm-written sections" in lower_message or "self-review" in lower_message:
        return {
            "phase": "llm_generation",
            "missing_inputs": ["Enough successful LLM-written sections and LLM self-review output"],
            "next_action": "Inspect LLM section errors and rerun with a working model configuration.",
            "next_command": _tcga_pipeline_rerun_command(args, example_root),
        }
    return {
        "phase": "draft_generation",
        "missing_inputs": ["Review draft-stage exception and generated logs"],
        "next_action": "Fix the draft-stage blocking error, then rerun tcga-pipeline.",
        "next_command": _tcga_pipeline_rerun_command(args, example_root),
    }


def _run_tcga_draft(args: argparse.Namespace) -> None:
    submission_grade = _apply_submission_grade_defaults(args)
    network_mode = _configure_network_mode(args, default_offline=True)
    compile_latex_requested = _configure_latex_compile(args)

    example_root = Path(args.example_root)
    baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
    code_path = example_root / "code" / "hyper-protosurv"
    if not code_path.is_dir():
        raise SystemExit(f"Hyper-ProtoSurv code directory not found: {code_path}")

    artifact_flow_summary_path, artifact_flow_summary = _load_tcga_artifact_flow_summary(args)
    if artifact_flow_summary_path and not getattr(args, "experiment_results", ""):
        experiment_path = _tcga_result_path_from_artifact_flow_summary(
            artifact_flow_summary_path,
            artifact_flow_summary,
        )
    else:
        experiment_path = _tcga_result_path(args, example_root)
    if artifact_flow_summary_path:
        print(f"TCGA artifact flow summary loaded from {artifact_flow_summary_path}")
    if not experiment_path.is_file():
        raise SystemExit(
            "TCGA experiment results file not found: "
            f"{experiment_path}. Create one with `paper-agent experiment-template "
            f"--output {experiment_path}` and replace every TODO with trained-model outputs."
        )
    experiment_results = experiment_path.read_text(encoding="utf-8")
    result_preflight = _validate_results_text(
        experiment_path,
        experiment_results,
        **_experiment_contract_kwargs(args),
        **_experiment_quality_kwargs(args, tcga_defaults=True),
        **_experiment_provenance_kwargs(args),
        **_experiment_artifact_consistency_kwargs(args),
    )
    if not _validated_results_are_strictly_acceptable(result_preflight):
        raise SystemExit("TCGA draft failed: experiment result validation failed in strict mode.")

    llm_client = None
    llm_mode = "disabled"
    effective_min_llm_sections = 0
    runtime_llm_config = load_llm_config()
    if args.disable_llm:
        os.environ["PAPER_AGENT_DISABLE_LLM"] = "1"
        runtime_llm_config = load_llm_config()
    else:
        os.environ["PAPER_AGENT_DISABLE_LLM"] = "0"
        config = load_llm_config()
        if not config.configured:
            raise SystemExit(
                "TCGA draft requires a configured LLM. Set DEEPSEEK_API_KEY or "
                "OPENAI_API_KEY and TEXT_MODEL, or pass --disable-llm for a deterministic run."
            )
        llm_client = LLMClient(config)
        _llm_preflight_check(llm_client, config, context="TCGA draft")
        llm_mode = "required"
        effective_min_llm_sections = args.min_llm_sections
        runtime_llm_config = config

    output_dir = Path(args.output_dir)
    project_name = args.project_name or _default_project_name(output_dir)
    skip_llm_self_review = args.disable_llm or args.skip_llm_self_review
    request = PaperRequest(
        project_name=project_name,
        target_venue=args.target_venue,
        baseline_pdf_path=str(baseline_pdf),
        code_path=str(code_path),
        template_zip_path=args.template_zip or None,
        template_dir_path=args.template_dir or None,
        experiment_results=experiment_results,
        keywords=[
            "whole-slide images",
            "survival prediction",
            "computational pathology",
            "hypergraph learning",
            *args.keyword,
        ],
        skip_llm_self_review=skip_llm_self_review,
    )
    state = PaperWorkflow(llm_client=llm_client).run(request)
    _record_runtime_modes(
        state,
        network_mode=network_mode,
        llm_mode=llm_mode,
        compile_latex_requested=compile_latex_requested,
        min_llm_sections=effective_min_llm_sections,
        llm_config=runtime_llm_config,
        submission_grade=submission_grade,
    )
    state.setdefault("artifacts", {})["experiment_results_source"] = "file"
    state["artifacts"]["experiment_results_path"] = str(experiment_path)
    _record_result_preflight(state, result_preflight)
    _record_tcga_artifact_flow_summary(state, artifact_flow_summary_path, artifact_flow_summary)
    SubmissionReadinessAgent().run(state)
    DraftReportAgent().run(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "draft.md"
    markdown_path.write_text(state["final_markdown"], encoding="utf-8")
    print(f"Markdown written to {markdown_path}")

    if args.zip:
        zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
        print(f"Overleaf zip written to {zip_path}")

    summary = _build_run_summary(state, markdown_path)
    acceptance_report_path = output_dir / "ACCEPTANCE_REPORT.md"
    summary["outputs"]["acceptance_report_path"] = str(acceptance_report_path)
    summary_path = _write_run_summary_data(summary, output_dir / "RUN_SUMMARY.json")
    _write_acceptance_report(
        summary,
        acceptance_report_path,
        min_llm_sections=effective_min_llm_sections,
        require_llm_self_review=not skip_llm_self_review,
    )
    print(f"Run summary written to {summary_path}")
    print(f"Acceptance report written to {acceptance_report_path}")

    artifacts = state.get("artifacts", {})
    successes = artifacts.get("section_writer_llm_successes", [])
    print(f"Section writer mode: {artifacts.get('section_writer_mode', 'unknown')}")
    print(f"LLM section successes: {len(successes)} ({', '.join(successes) or 'none'})")
    print(f"Evidence guard findings: {len(artifacts.get('evidence_guard_findings', []))}")
    print(f"Review findings: {len(state.get('review_findings', []))}")
    print(f"Template source: {state['venue_template'].template_source}")
    print(f"Bibliography entries: {len(state.get('bibliography', []))}")
    print(f"Network mode: {network_mode}")
    print(f"LLM mode: {llm_mode}")
    print(f"LLM self-review: {_llm_self_review_mode(state)}")
    print(f"Submission grade: {submission_grade}")
    print(f"LaTeX written to {state['latex_output_path']}")
    if len(successes) < effective_min_llm_sections:
        raise SystemExit(
            f"TCGA draft failed: expected at least {effective_min_llm_sections} "
            f"LLM-written sections, got {len(successes)}."
        )
    if not skip_llm_self_review and _llm_self_review_mode(state) != "llm":
        raise SystemExit("TCGA draft failed: LLM self-review did not complete in llm mode.")
    print("TCGA draft run completed.")


def _run_llm_draft_smoke(args: argparse.Namespace) -> None:
    network_mode = _configure_network_mode(args, default_offline=True)
    compile_latex_requested = _configure_latex_compile(args)
    config = load_llm_config()
    if not config.configured:
        raise SystemExit(
            "LLM draft smoke requires a configured LLM. Set DEEPSEEK_API_KEY or "
            "OPENAI_API_KEY and TEXT_MODEL, and do not set PAPER_AGENT_DISABLE_LLM=1."
        )
    example_root = Path(args.example_root)
    baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
    code_path = example_root / "code" / "hyper-protosurv"
    if not code_path.is_dir():
        raise SystemExit(f"Hyper-ProtoSurv code directory not found: {code_path}")

    experiment_path = _resolve_project_relative_path(args.experiment_results)
    if not experiment_path.is_file():
        raise SystemExit(f"Experiment results file not found: {experiment_path}")
    experiment_results = experiment_path.read_text(encoding="utf-8")
    result_preflight = _validate_results_text(
        experiment_path,
        experiment_results,
        **_experiment_contract_kwargs(args),
        **_experiment_provenance_kwargs(args),
        **_experiment_artifact_consistency_kwargs(args),
    )
    if args.strict_results and not _validated_results_are_strictly_acceptable(result_preflight):
        raise SystemExit("LLM draft smoke failed: experiment result validation failed in strict mode.")
    llm_client = LLMClient(config)
    _llm_preflight_check(llm_client, config, context="LLM draft smoke")

    output_dir = Path(args.output_dir)
    project_name = args.project_name or _default_project_name(output_dir)
    request = PaperRequest(
        project_name=project_name,
        target_venue=args.target_venue,
        baseline_pdf_path=str(baseline_pdf),
        code_path=str(code_path),
        experiment_results=experiment_results,
        keywords=[
            "whole-slide images",
            "survival prediction",
            "computational pathology",
            "hypergraph learning",
        ],
        skip_llm_self_review=not args.include_llm_self_review,
    )
    state = PaperWorkflow(llm_client=llm_client).run(request)
    _record_runtime_modes(
        state,
        network_mode=network_mode,
        llm_mode="required",
        compile_latex_requested=compile_latex_requested,
        min_llm_sections=args.min_llm_sections,
        llm_config=config,
    )
    state.setdefault("artifacts", {})["experiment_results_source"] = "file"
    state["artifacts"]["experiment_results_path"] = str(experiment_path)
    _record_result_preflight(state, result_preflight)
    SubmissionReadinessAgent().run(state)
    DraftReportAgent().run(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "draft.md"
    markdown_path.write_text(state["final_markdown"], encoding="utf-8")
    print(f"Markdown written to {markdown_path}")

    if args.zip:
        zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
        print(f"Overleaf zip written to {zip_path}")

    summary = _build_run_summary(state, markdown_path)
    acceptance_report_path = output_dir / "ACCEPTANCE_REPORT.md"
    summary["outputs"]["acceptance_report_path"] = str(acceptance_report_path)
    summary_path = _write_run_summary_data(summary, output_dir / "RUN_SUMMARY.json")
    _write_acceptance_report(
        summary,
        acceptance_report_path,
        min_llm_sections=args.min_llm_sections,
        require_llm_self_review=args.include_llm_self_review,
    )
    print(f"Run summary written to {summary_path}")
    print(f"Acceptance report written to {acceptance_report_path}")

    artifacts = state.get("artifacts", {})
    successes = artifacts.get("section_writer_llm_successes", [])
    errors = artifacts.get("section_writer_section_errors", {})
    print(f"Section writer mode: {artifacts.get('section_writer_mode', 'unknown')}")
    print(f"LLM section successes: {len(successes)} ({', '.join(successes) or 'none'})")
    if errors:
        print(f"LLM section errors: {len(errors)}")
    print(f"Evidence guard findings: {len(artifacts.get('evidence_guard_findings', []))}")
    print(f"Review findings: {len(state.get('review_findings', []))}")
    print(f"LLM self-review: {_llm_self_review_mode(state)}")
    if len(successes) < args.min_llm_sections:
        raise SystemExit(
            f"LLM draft smoke failed: expected at least {args.min_llm_sections} "
            f"LLM-written sections, got {len(successes)}."
        )
    if args.include_llm_self_review and _llm_self_review_mode(state) != "llm":
        raise SystemExit("LLM draft smoke failed: LLM self-review did not complete in llm mode.")
    print("LLM draft smoke passed.")


def _run_paper_e2e_smoke(args: argparse.Namespace) -> None:
    if args.include_llm_self_review and not args.require_llm:
        raise SystemExit(
            "paper-e2e-smoke requires --require-llm when --include-llm-self-review is set."
        )
    if args.min_llm_sections and not args.require_llm:
        raise SystemExit(
            "paper-e2e-smoke requires --require-llm when --min-llm-sections is greater than zero."
        )

    network_mode = _configure_network_mode(args, default_offline=True)
    compile_latex_requested = _configure_latex_compile(args)
    output_dir = Path(args.output_dir)
    baseline_pdf = _resolve_baseline_pdf(args.baseline_pdf)
    code_path = _resolve_project_relative_path(args.code_path)
    if not code_path.is_dir():
        raise SystemExit(f"Project code directory not found: {code_path}")

    experiment_path = _resolve_project_relative_path(args.experiment_results)
    generated_result_summary = None
    if args.generate_results_from_artifacts:
        generated_result_summary = _paper_e2e_generate_results_from_artifacts(args, experiment_path)
    if not experiment_path.is_file():
        raise SystemExit(f"Experiment results file not found: {experiment_path}")
    experiment_results = experiment_path.read_text(encoding="utf-8")
    result_preflight = _validate_results_text(
        experiment_path,
        experiment_results,
        **_experiment_contract_kwargs(args),
        **_experiment_provenance_kwargs(args),
        **_experiment_artifact_consistency_kwargs(args),
    )
    if args.strict_results and not _validated_results_are_strictly_acceptable(result_preflight):
        summary_path = _write_paper_e2e_smoke_failure_summary(
            args,
            baseline_pdf=baseline_pdf,
            code_path=code_path,
            experiment_path=experiment_path,
            result_preflight=result_preflight,
            reason="experiment result validation failed in strict mode",
        )
        print(f"Run summary written to {summary_path}")
        raise SystemExit(
            "paper-e2e-smoke failed: experiment result validation failed in strict mode. "
            f"Summary: {summary_path}"
        )

    runtime_llm_config = load_llm_config()
    llm_client = None
    llm_mode = "disabled"
    llm_preflight_summary: dict[str, object] = {}
    if args.require_llm:
        os.environ["PAPER_AGENT_DISABLE_LLM"] = "0"
        runtime_llm_config = load_llm_config()
        if not runtime_llm_config.configured:
            summary_path = _write_paper_e2e_smoke_llm_failure_summary(
                args,
                baseline_pdf=baseline_pdf,
                code_path=code_path,
                experiment_path=experiment_path,
                result_preflight=result_preflight,
                llm_config=runtime_llm_config,
                raw_error=(
                    "paper-e2e-smoke requires a configured LLM. Set DEEPSEEK_API_KEY or "
                    "OPENAI_API_KEY and TEXT_MODEL, or run without --require-llm for a deterministic smoke."
                ),
            )
            print(f"Run summary written to {summary_path}")
            print(
                "Acceptance report written to "
                f"{Path(args.output_dir) / 'ACCEPTANCE_REPORT.md'}"
            )
            raise SystemExit(
                "paper-e2e-smoke failed: configured LLM is required. "
                f"Summary: {summary_path}"
            )
        llm_client = LLMClient(runtime_llm_config)
        try:
            llm_preflight_result = _llm_preflight_check(
                llm_client,
                runtime_llm_config,
                context="paper-e2e-smoke",
            )
        except SystemExit as exc:
            summary_path = _write_paper_e2e_smoke_llm_failure_summary(
                args,
                baseline_pdf=baseline_pdf,
                code_path=code_path,
                experiment_path=experiment_path,
                result_preflight=result_preflight,
                llm_config=runtime_llm_config,
                raw_error=str(exc),
            )
            print(f"Run summary written to {summary_path}")
            print(
                "Acceptance report written to "
                f"{Path(args.output_dir) / 'ACCEPTANCE_REPORT.md'}"
            )
            raise SystemExit(
                "paper-e2e-smoke failed: LLM preflight failed. "
                f"Summary: {summary_path}"
            ) from exc
        llm_preflight_summary = {"status": "pass", **llm_preflight_result}
        llm_mode = "required"
    else:
        os.environ["PAPER_AGENT_DISABLE_LLM"] = "1"
        runtime_llm_config = load_llm_config()

    project_name = args.project_name or _default_project_name(output_dir)
    request = PaperRequest(
        project_name=project_name,
        target_venue=args.target_venue,
        baseline_pdf_path=str(baseline_pdf),
        code_path=str(code_path),
        template_zip_path=args.template_zip or None,
        template_dir_path=args.template_dir or None,
        experiment_results=experiment_results,
        keywords=list(args.keyword or []),
        skip_llm_self_review=not args.include_llm_self_review,
    )
    state = PaperWorkflow(llm_client=llm_client).run(request)
    _record_runtime_modes(
        state,
        network_mode=network_mode,
        llm_mode=llm_mode,
        compile_latex_requested=compile_latex_requested,
        min_llm_sections=args.min_llm_sections if args.require_llm else 0,
        llm_config=runtime_llm_config,
    )
    artifacts = state.setdefault("artifacts", {})
    artifacts["experiment_results_source"] = "file"
    artifacts["experiment_results_path"] = str(experiment_path)
    artifacts["paper_e2e_smoke_inputs"] = {
        "baseline_pdf": str(baseline_pdf),
        "code_path": str(code_path),
        "experiment_results": str(experiment_path),
        "target_venue": args.target_venue,
    }
    if llm_preflight_summary:
        artifacts["paper_e2e_llm_preflight"] = llm_preflight_summary
    if generated_result_summary:
        artifacts["paper_e2e_generated_results_from_artifacts"] = generated_result_summary
    _record_result_preflight(state, result_preflight)
    SubmissionReadinessAgent().run(state)
    DraftReportAgent().run(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "draft.md"
    markdown_path.write_text(state["final_markdown"], encoding="utf-8")
    print(f"Markdown written to {markdown_path}")

    zip_path = Path(args.zip) if args.zip else None
    if zip_path:
        zip_path = _write_latex_zip_and_refresh(state, zip_path)
        print(f"Overleaf zip written to {zip_path}")

    summary = _build_run_summary(state, markdown_path)
    acceptance_report_path = output_dir / "ACCEPTANCE_REPORT.md"
    summary["outputs"]["acceptance_report_path"] = str(acceptance_report_path)
    summary["smoke_contract"] = _paper_e2e_smoke_contract(
        args,
        baseline_pdf=baseline_pdf,
        code_path=code_path,
        experiment_path=experiment_path,
        markdown_path=markdown_path,
        summary_path=output_dir / "RUN_SUMMARY.json",
        acceptance_report_path=acceptance_report_path,
        zip_path=zip_path,
        result_preflight=result_preflight,
        llm_mode=llm_mode,
        llm_preflight_status=str(llm_preflight_summary.get("status", "not_recorded")),
        generated_results_from_artifacts=bool(args.generate_results_from_artifacts),
        status="pass",
    )
    summary_path = _write_run_summary_data(summary, output_dir / "RUN_SUMMARY.json")
    _write_acceptance_report(
        summary,
        acceptance_report_path,
        min_llm_sections=args.min_llm_sections if args.require_llm else 0,
        require_llm_self_review=args.include_llm_self_review,
    )
    print(f"Run summary written to {summary_path}")
    print(f"Acceptance report written to {acceptance_report_path}")

    successes = artifacts.get("section_writer_llm_successes", [])
    print(f"Input baseline PDF: {baseline_pdf}")
    print(f"Input code path: {code_path}")
    print(f"Input experiment results: {experiment_path}")
    print(f"Target venue: {args.target_venue}")
    print(f"LLM mode: {llm_mode}")
    print(f"LLM section successes: {len(successes)} ({', '.join(successes) or 'none'})")
    if len(successes) < (args.min_llm_sections if args.require_llm else 0):
        raise SystemExit(
            f"paper-e2e-smoke failed: expected at least {args.min_llm_sections} "
            f"LLM-written sections, got {len(successes)}."
        )
    if args.include_llm_self_review and _llm_self_review_mode(state) != "llm":
        raise SystemExit("paper-e2e-smoke failed: LLM self-review did not complete in llm mode.")
    print("paper-e2e-smoke passed.")


def _paper_e2e_smoke_contract(
    args: argparse.Namespace,
    *,
    baseline_pdf: Path,
    code_path: Path,
    experiment_path: Path,
    markdown_path: Path,
    summary_path: Path,
    acceptance_report_path: Path,
    zip_path: Path | None,
    result_preflight: dict[str, object],
    llm_mode: str,
    llm_preflight_status: str = "not_recorded",
    generated_results_from_artifacts: bool = False,
    status: str = "pass",
) -> dict[str, object]:
    return {
        "schema_version": "paper-e2e-smoke/v1",
        "status": status,
        "required_inputs": {
            "baseline_pdf": str(baseline_pdf),
            "code_path": str(code_path),
            "experiment_results": str(experiment_path),
            "target_venue": args.target_venue,
        },
        "outputs": {
            "markdown": str(markdown_path),
            "summary": str(summary_path),
            "acceptance_report": str(acceptance_report_path),
            "zip": str(zip_path or ""),
        },
        "checks": {
            "baseline_pdf_exists": baseline_pdf.is_file(),
            "code_path_exists": code_path.is_dir(),
            "experiment_results_exists": experiment_path.is_file(),
            "strict_results": bool(args.strict_results),
            "strict_results_accepted": _validated_results_are_strictly_acceptable(result_preflight),
            "generated_results_from_artifacts": generated_results_from_artifacts,
            "artifacts_dir": str(_paper_e2e_completed_artifacts_dir(args, experiment_path))
            if generated_results_from_artifacts
            else "",
            "llm_mode": llm_mode,
            "llm_preflight_status": llm_preflight_status,
            "min_llm_sections": int(args.min_llm_sections if args.require_llm else 0),
        },
    }


def _write_paper_e2e_smoke_failure_summary(
    args: argparse.Namespace,
    *,
    baseline_pdf: Path,
    code_path: Path,
    experiment_path: Path,
    result_preflight: dict[str, object],
    reason: str,
) -> Path:
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "RUN_SUMMARY.json"
    acceptance_report_path = output_dir / "ACCEPTANCE_REPORT.md"
    zip_path = Path(args.zip) if args.zip else None
    blocking_items = _paper_e2e_result_preflight_issues(result_preflight)
    if not blocking_items:
        blocking_items = [reason]
    artifact_template = _paper_e2e_maybe_write_artifact_template(args, experiment_path)
    if artifact_template.get("status") == "written":
        print(f"Artifact templates written to {artifact_template.get('output_dir', '')}")
    summary = {
        "status": "blocked",
        "pipeline_phase": "paper_e2e_smoke_preflight",
        "project_name": args.project_name or _default_project_name(output_dir),
        "target_venue": args.target_venue,
        "blocking_items": blocking_items,
        "missing_inputs": [],
        "next_action": (
            "Repair the experiment result file until strict checks pass, or rerun with "
            "--no-strict-results for a toolchain-only smoke."
        ),
        "next_command": _paper_e2e_validate_results_command(args, experiment_path),
        "next_actions": _paper_e2e_smoke_failure_next_actions(args, experiment_path),
        "artifact_template": artifact_template,
        "experiment_evidence": result_preflight.get("experiment_evidence", {}),
        "experiment_contract": result_preflight.get("experiment_contract", {}),
        "experiment_quality": result_preflight.get("experiment_quality", {}),
        "experiment_provenance": result_preflight.get("experiment_provenance", {}),
        "experiment_artifact_consistency": result_preflight.get(
            "experiment_artifact_consistency",
            {},
        ),
        "smoke_contract": _paper_e2e_smoke_contract(
            args,
            baseline_pdf=baseline_pdf,
            code_path=code_path,
            experiment_path=experiment_path,
            markdown_path=output_dir / "draft.md",
            summary_path=summary_path,
            acceptance_report_path=acceptance_report_path,
            zip_path=zip_path,
            result_preflight=result_preflight,
            llm_mode="not_started",
            generated_results_from_artifacts=bool(
                getattr(args, "generate_results_from_artifacts", False)
            ),
            status="blocked",
        ),
    }
    return _write_run_summary_data(summary, summary_path)


def _write_paper_e2e_smoke_llm_failure_summary(
    args: argparse.Namespace,
    *,
    baseline_pdf: Path,
    code_path: Path,
    experiment_path: Path,
    result_preflight: dict[str, object],
    llm_config: LLMConfig,
    raw_error: str,
) -> Path:
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "RUN_SUMMARY.json"
    acceptance_report_path = output_dir / "ACCEPTANCE_REPORT.md"
    zip_path = Path(args.zip) if args.zip else None
    diagnostics = _llm_failure_diagnostics(llm_config, raw_error)
    diagnosis = str(diagnostics.get("diagnosis", "LLM preflight failed."))
    next_action = "Fix LLM configuration or provider quota before rerunning LLM-required paper smoke."
    summary = {
        "status": "blocked",
        "pipeline_phase": "paper_e2e_smoke_llm_preflight",
        "project_name": args.project_name or _default_project_name(output_dir),
        "target_venue": args.target_venue,
        "blocking_items": [f"llm: {diagnosis}"],
        "missing_inputs": [] if llm_config.configured else ["configured LLM API key/model"],
        "next_action": next_action,
        "next_command": "paper-agent llm-doctor --summary outputs/llm-doctor.json",
        "next_actions": [
            {
                "category": "llm",
                "action": next_action,
                "command": "paper-agent llm-doctor --summary outputs/llm-doctor.json",
            },
            {
                "category": "paper_e2e_smoke",
                "action": "Rerun the explicit input-to-paper smoke after LLM repair.",
                "command": _paper_e2e_smoke_rerun_command(args),
            },
        ],
        "outputs": {
            "summary": str(summary_path),
            "acceptance_report": str(acceptance_report_path),
            "markdown": str(output_dir / "draft.md"),
            "zip": str(zip_path or ""),
        },
        "llm_diagnostics": diagnostics,
        "experiment_evidence": result_preflight.get("experiment_evidence", {}),
        "experiment_contract": result_preflight.get("experiment_contract", {}),
        "experiment_quality": result_preflight.get("experiment_quality", {}),
        "experiment_provenance": result_preflight.get("experiment_provenance", {}),
        "experiment_artifact_consistency": result_preflight.get(
            "experiment_artifact_consistency",
            {},
        ),
        "smoke_contract": _paper_e2e_smoke_contract(
            args,
            baseline_pdf=baseline_pdf,
            code_path=code_path,
            experiment_path=experiment_path,
            markdown_path=output_dir / "draft.md",
            summary_path=summary_path,
            acceptance_report_path=acceptance_report_path,
            zip_path=zip_path,
            result_preflight=result_preflight,
            llm_mode="failed_preflight",
            llm_preflight_status="fail",
            generated_results_from_artifacts=bool(
                getattr(args, "generate_results_from_artifacts", False)
            ),
            status="blocked",
        ),
    }
    _write_paper_e2e_blocked_acceptance_report(summary, acceptance_report_path)
    return _write_run_summary_data(summary, summary_path)


def _write_paper_e2e_blocked_acceptance_report(summary: dict[str, object], report_path: Path) -> Path:
    diagnostics = summary.get("llm_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    smoke_contract = summary.get("smoke_contract", {})
    checks = smoke_contract.get("checks", {}) if isinstance(smoke_contract, dict) else {}
    next_actions = summary.get("next_actions", [])
    if not isinstance(next_actions, list):
        next_actions = []
    lines = [
        "# Paper Agent Blocked Acceptance Report",
        "",
        f"- Status: {summary.get('status', 'blocked')}",
        f"- Pipeline phase: {summary.get('pipeline_phase', '')}",
        f"- Project: {summary.get('project_name', '')}",
        f"- Target venue: {summary.get('target_venue', '')}",
        "",
        "## Input Contract",
        "",
        f"- Baseline PDF exists: {checks.get('baseline_pdf_exists', False)}",
        f"- Code path exists: {checks.get('code_path_exists', False)}",
        f"- Experiment results exists: {checks.get('experiment_results_exists', False)}",
        f"- Strict results accepted: {checks.get('strict_results_accepted', False)}",
        f"- LLM mode: {checks.get('llm_mode', '')}",
        f"- LLM preflight status: {checks.get('llm_preflight_status', '')}",
        "",
        "## LLM Diagnostics",
        "",
        f"- Provider: {diagnostics.get('provider', '')}",
        f"- Model: {diagnostics.get('model', '')}",
        f"- Endpoint host: {diagnostics.get('endpoint_host', '')}",
        f"- Failure kind: {diagnostics.get('failure_kind', '')}",
        f"- Diagnosis: {diagnostics.get('diagnosis', '')}",
        f"- Raw provider error: {diagnostics.get('raw_error', '')}",
        "",
        "## Next Actions",
        "",
        "| Category | Action | Command |",
        "|---|---|---|",
    ]
    for action in next_actions:
        if not isinstance(action, dict):
            continue
        lines.append(
            "| "
            f"{_table_safe(str(action.get('category', '')))} | "
            f"{_table_safe(str(action.get('action', '')))} | "
            f"`{_table_safe(str(action.get('command', '')))}` |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _paper_e2e_maybe_write_artifact_template(
    args: argparse.Namespace,
    experiment_path: Path,
) -> dict[str, object]:
    output_dir = _paper_e2e_artifact_template_dir(args, experiment_path)
    if not getattr(args, "write_artifact_template", False):
        return {
            "status": "skipped",
            "output_dir": str(output_dir),
            "command": _paper_e2e_artifact_template_command(output_dir),
        }

    datasets = list(getattr(args, "artifact_template_dataset", []) or _tcga_default_datasets())
    style = str(getattr(args, "artifact_template_style", "long"))
    method = str(getattr(args, "artifact_method", "Hyper-ProtoSurv ours"))
    baseline = str(getattr(args, "artifact_baseline", "ProtoSurv baseline"))
    metric = str(getattr(args, "artifact_metric", "C-index"))
    seed = str(getattr(args, "artifact_template_seed", "2026"))
    files = _tcga_artifact_template_bundle(
        style=style,
        datasets=datasets,
        method=method,
        baseline=baseline,
        metric=metric,
        seed=seed,
    )
    try:
        written_paths = _write_tcga_artifact_template_bundle(
            output_dir,
            files,
            force=bool(getattr(args, "artifact_template_force", False)),
        )
    except SystemExit as exc:
        return {
            "status": "failed",
            "output_dir": str(output_dir),
            "error": str(exc),
            "command": _paper_e2e_artifact_template_command(output_dir),
        }
    return {
        "status": "written",
        "output_dir": str(output_dir),
        "files": [str(path) for path in written_paths],
        "datasets": datasets,
        "style": style,
        "contains_todo": True,
        "command": _paper_e2e_artifact_template_command(output_dir),
    }


def _paper_e2e_generate_results_from_artifacts(
    args: argparse.Namespace,
    experiment_path: Path,
) -> dict[str, object]:
    artifact_dir = _paper_e2e_completed_artifacts_dir(args, experiment_path)
    result_args = argparse.Namespace(
        example_root=str(experiment_path.parent.parent),
        artifacts_dir=str(artifact_dir),
        main_csv="",
        ablation_csv="",
        sensitivity_csv="",
        stats_csv="",
        output=str(experiment_path),
        method=str(getattr(args, "artifact_method", "Hyper-ProtoSurv ours")),
        baseline=str(getattr(args, "artifact_baseline", "ProtoSurv baseline")),
        metric=str(getattr(args, "artifact_metric", "C-index")),
        dataset=list(getattr(args, "artifact_template_dataset", []) or []),
        strict=True,
        require_ablation=bool(getattr(args, "require_ablation", True)),
        require_sensitivity=bool(getattr(args, "require_sensitivity", True)),
        require_statistical_tests=bool(getattr(args, "require_statistical_tests", True)),
    )
    summary = _run_tcga_results_from_artifacts(result_args) or {}
    return {
        "status": "generated",
        "artifacts_dir": str(artifact_dir),
        "experiment_results": str(experiment_path),
        "experiment_contract_status": str(
            summary.get("experiment_contract", {}).get("status", "unknown")
        )
        if isinstance(summary.get("experiment_contract", {}), dict)
        else "unknown",
        "experiment_provenance_status": str(
            summary.get("experiment_provenance", {}).get("status", "unknown")
        )
        if isinstance(summary.get("experiment_provenance", {}), dict)
        else "unknown",
        "experiment_artifact_consistency_status": str(
            summary.get("experiment_artifact_consistency", {}).get("status", "unknown")
        )
        if isinstance(summary.get("experiment_artifact_consistency", {}), dict)
        else "unknown",
    }


def _paper_e2e_smoke_failure_next_actions(
    args: argparse.Namespace,
    experiment_path: Path,
) -> list[dict[str, str]]:
    artifact_dir = _paper_e2e_artifact_template_dir(args, experiment_path)
    return [
        {
            "category": "validate_results",
            "action": "Inspect strict result validation errors.",
            "command": _paper_e2e_validate_results_command(args, experiment_path),
        },
        {
            "category": "result_artifacts",
            "action": (
                "If result CSV artifacts do not exist yet, create the export templates and fill every "
                "TODO with real trained-model outputs."
            ),
            "command": _paper_e2e_artifact_template_command(artifact_dir),
        },
        {
            "category": "experiment_results",
            "action": "Generate a strict result Markdown file from completed result CSV artifacts.",
            "command": _paper_e2e_results_from_artifacts_command(artifact_dir, experiment_path),
        },
        {
            "category": "paper_e2e_smoke",
            "action": "Rerun the explicit input-to-paper smoke after result repair.",
            "command": _paper_e2e_smoke_rerun_command(args),
        },
    ]


def _paper_e2e_result_preflight_issues(result_preflight: dict[str, object]) -> list[str]:
    issues: list[str] = []
    sections = (
        ("experiment_contract", "contract"),
        ("experiment_quality", "quality"),
        ("experiment_provenance", "provenance"),
        ("experiment_artifact_consistency", "artifact_consistency"),
    )
    for section_key, label in sections:
        section = result_preflight.get(section_key, {})
        if not isinstance(section, dict):
            continue
        for error in section.get("errors", []) or []:
            issues.append(f"{label}: {error}")
    if not issues and not _validated_results_are_strictly_acceptable(result_preflight):
        issues.append("experiment result file did not satisfy strict acceptance checks")
    return issues


def _paper_e2e_artifact_template_command(artifact_dir: Path) -> str:
    parts = [
        "paper-agent",
        "tcga-artifact-template",
        "--output-dir",
        str(artifact_dir),
    ]
    return " ".join(_powershell_arg(part) for part in parts)


def _paper_e2e_artifact_template_dir(args: argparse.Namespace, experiment_path: Path) -> Path:
    path_value = str(getattr(args, "artifact_template_dir", "") or "")
    if path_value:
        return _resolve_project_relative_path(path_value)
    return experiment_path.parent / "logs"


def _paper_e2e_completed_artifacts_dir(args: argparse.Namespace, experiment_path: Path) -> Path:
    path_value = str(getattr(args, "artifacts_dir", "") or "")
    if path_value:
        return _resolve_project_relative_path(path_value)
    return _paper_e2e_artifact_template_dir(args, experiment_path)


def _paper_e2e_results_from_artifacts_command(artifact_dir: Path, experiment_path: Path) -> str:
    parts = [
        "paper-agent",
        "tcga-results-from-artifacts",
        "--artifacts-dir",
        str(artifact_dir),
        "--output",
        str(experiment_path),
        "--strict",
    ]
    return " ".join(_powershell_arg(part) for part in parts)


def _paper_e2e_validate_results_command(args: argparse.Namespace, experiment_path: Path) -> str:
    parts = [
        "paper-agent",
        "validate-results",
        "--experiment-results",
        str(experiment_path),
        "--strict",
    ]
    for flag in ("ablation", "sensitivity", "statistical-tests"):
        attr = f"require_{flag.replace('-', '_')}"
        if not bool(getattr(args, attr, True)):
            parts.append(f"--no-require-{flag}")
    if getattr(args, "require_provenance", False):
        parts.append("--require-provenance")
    if getattr(args, "require_artifact_consistency", False):
        parts.append("--require-artifact-consistency")
    return " ".join(_powershell_arg(part) for part in parts)


def _paper_e2e_smoke_rerun_command(args: argparse.Namespace) -> str:
    parts = [
        "paper-agent",
        "paper-e2e-smoke",
        "--baseline-pdf",
        args.baseline_pdf,
        "--code-path",
        args.code_path,
        "--experiment-results",
        args.experiment_results,
        "--target-venue",
        args.target_venue,
        "--output-dir",
        args.output_dir,
    ]
    if args.project_name:
        parts.extend(["--project-name", args.project_name])
    if args.zip:
        parts.extend(["--zip", args.zip])
    for keyword in args.keyword or []:
        parts.extend(["--keyword", keyword])
    if args.template_zip:
        parts.extend(["--template-zip", args.template_zip])
    if args.template_dir:
        parts.extend(["--template-dir", args.template_dir])
    if args.online:
        parts.append("--online")
    if args.offline:
        parts.append("--offline")
    if args.require_llm:
        parts.append("--require-llm")
    if args.min_llm_sections:
        parts.extend(["--min-llm-sections", str(args.min_llm_sections)])
    if args.include_llm_self_review:
        parts.append("--include-llm-self-review")
    if args.compile_latex:
        parts.append("--compile-latex")
    parts.append("--strict-results" if args.strict_results else "--no-strict-results")
    if getattr(args, "write_artifact_template", False):
        parts.append("--write-artifact-template")
    if getattr(args, "artifact_template_dir", ""):
        parts.extend(["--artifact-template-dir", args.artifact_template_dir])
    if getattr(args, "artifact_template_force", False):
        parts.append("--artifact-template-force")
    style = getattr(args, "artifact_template_style", "long")
    if style != "long":
        parts.extend(["--artifact-template-style", style])
    seed = getattr(args, "artifact_template_seed", "2026")
    if seed != "2026":
        parts.extend(["--artifact-template-seed", seed])
    for dataset in getattr(args, "artifact_template_dataset", []) or []:
        parts.extend(["--artifact-template-dataset", dataset])
    artifact_method = getattr(args, "artifact_method", "Hyper-ProtoSurv ours")
    if artifact_method != "Hyper-ProtoSurv ours":
        parts.extend(["--artifact-method", artifact_method])
    artifact_baseline = getattr(args, "artifact_baseline", "ProtoSurv baseline")
    if artifact_baseline != "ProtoSurv baseline":
        parts.extend(["--artifact-baseline", artifact_baseline])
    artifact_metric = getattr(args, "artifact_metric", "C-index")
    if artifact_metric != "C-index":
        parts.extend(["--artifact-metric", artifact_metric])
    if getattr(args, "generate_results_from_artifacts", False):
        parts.append("--generate-results-from-artifacts")
    if getattr(args, "artifacts_dir", ""):
        parts.extend(["--artifacts-dir", args.artifacts_dir])
    for flag in ("ablation", "sensitivity", "statistical-tests"):
        attr = f"require_{flag.replace('-', '_')}"
        if not bool(getattr(args, attr, True)):
            parts.append(f"--no-require-{flag}")
    if getattr(args, "require_provenance", False):
        parts.append("--require-provenance")
    if getattr(args, "require_artifact_consistency", False):
        parts.append("--require-artifact-consistency")
    return " ".join(_powershell_arg(part) for part in parts)


def _resolve_project_relative_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _project_root() / path


def _default_project_name(output_dir: Path) -> str:
    name = output_dir.name or "paper"
    if name.lower() not in {"out", "output", "outputs", "result", "results", "draft", "paper"}:
        return name
    parent = output_dir.parent.name
    if parent and parent not in {".", name}:
        return f"{parent}-{name}"
    return name


def _validate_results_file(
    path: Path,
    summary_path: Path | None = None,
    *,
    source: str = "file",
    require_ablation: bool = True,
    require_sensitivity: bool = True,
    require_statistical_tests: bool = True,
    expected_datasets: list[str] | None = None,
    expected_metrics: list[str] | None = None,
    expected_method: str = "",
    expected_baseline: str = "",
    require_provenance: bool = False,
    require_artifact_consistency: bool = False,
) -> dict:
    path = _resolve_project_relative_path(str(path))
    if not path.is_file():
        raise SystemExit(f"Experiment results file not found: {path}")

    raw = path.read_text(encoding="utf-8")
    return _validate_results_text(
        path,
        raw,
        summary_path=summary_path,
        source=source,
        require_ablation=require_ablation,
        require_sensitivity=require_sensitivity,
        require_statistical_tests=require_statistical_tests,
        expected_datasets=expected_datasets,
        expected_metrics=expected_metrics,
        expected_method=expected_method,
        expected_baseline=expected_baseline,
        require_provenance=require_provenance,
        require_artifact_consistency=require_artifact_consistency,
    )


def _validate_results_text(
    path: Path,
    raw: str,
    summary_path: Path | None = None,
    *,
    source: str = "file",
    require_ablation: bool = True,
    require_sensitivity: bool = True,
    require_statistical_tests: bool = True,
    expected_datasets: list[str] | None = None,
    expected_metrics: list[str] | None = None,
    expected_method: str = "",
    expected_baseline: str = "",
    require_provenance: bool = False,
    require_artifact_consistency: bool = False,
) -> dict:
    state = ExperimentAnalyzerAgent().run(
        {
            "request": PaperRequest(
                project_name="validate-results",
                target_venue="unspecified",
                experiment_results=raw,
            )
        }
    )
    experiments = state["experiments"]
    contract = validate_experiment_contract(
        experiments,
        require_ablation=require_ablation,
        require_sensitivity=require_sensitivity,
        require_statistical_tests=require_statistical_tests,
    )
    evidence = classify_experiment_evidence(
        source=source,
        path=str(path),
        text=raw,
        result_table_count=len(experiments.result_tables),
    )
    quality = assess_experiment_quality(
        experiments,
        expected_datasets=expected_datasets,
        expected_metrics=expected_metrics,
        expected_method=expected_method,
        expected_baseline=expected_baseline,
    )
    provenance = assess_experiment_provenance(
        raw,
        result_path=path,
        require_provenance=require_provenance,
    )
    artifact_consistency = assess_experiment_artifact_consistency(
        experiments,
        provenance,
        require_consistency=require_artifact_consistency,
    )
    summary = {
        "path": str(path),
        "source": source,
        "experiment_evidence": evidence,
        "experiment_contract": contract,
        "experiment_contract_requirements": contract.get("requirements", {}),
        "experiment_quality": quality,
        "experiment_provenance": provenance,
        "experiment_artifact_consistency": artifact_consistency,
        "datasets": experiments.datasets,
        "metrics": experiments.metrics,
        "missing_details": experiments.missing_details,
        "observations": experiments.observations,
    }

    checks = contract.get("checks", {})
    print(f"Experiment results: {path}")
    print(f"Experiment evidence kind: {evidence.get('kind', 'unknown')}")
    print(f"Experiment result contract: {contract.get('status', 'unknown')}")
    requirements = contract.get("requirements", {})
    print(
        "Requirements: "
        f"ablation={requirements.get('ablation', True)}; "
        f"sensitivity={requirements.get('sensitivity', True)}; "
        f"statistical_tests={requirements.get('statistical_tests', True)}"
    )
    print(
        "Coverage: "
        f"main={checks.get('result_tables', 0)}; "
        f"comparisons={checks.get('numeric_comparisons', 0)}; "
        f"datasets={checks.get('datasets', 0)}; "
        f"metrics={checks.get('metrics', 0)}; "
        f"ablation={checks.get('ablation_items', 0)}; "
        f"sensitivity={checks.get('sensitivity_items', 0)}; "
        f"statistical={checks.get('statistical_tests', 0)}"
    )
    for error in contract.get("errors", []):
        print(f"ERROR: {error}")
    for warning in contract.get("warnings", []):
        print(f"WARNING: {warning}")
    if quality.get("status") != "not_configured":
        print(f"Experiment result quality: {quality.get('status', 'unknown')}")
        for error in quality.get("errors", []):
            print(f"QUALITY ERROR: {error}")
        for warning in quality.get("warnings", []):
            print(f"QUALITY WARNING: {warning}")
    print(f"Experiment result provenance: {provenance.get('status', 'unknown')}")
    provenance_checks = provenance.get("checks", {})
    if isinstance(provenance_checks, dict):
        print(
            "Provenance fingerprints: "
            f"{provenance_checks.get('fingerprinted_local_paths', 0)}/"
            f"{provenance_checks.get('local_paths', 0)} local files; "
            f"verified_checksums={provenance_checks.get('verified_checksums', 0)}; "
            f"checksum_mismatches={provenance_checks.get('checksum_mismatches', 0)}"
        )
    for error in provenance.get("errors", []):
        print(f"PROVENANCE ERROR: {error}")
    for warning in provenance.get("warnings", []):
        print(f"PROVENANCE WARNING: {warning}")
    if artifact_consistency.get("status") != "not_configured":
        consistency_checks = artifact_consistency.get("checks", {})
        if not isinstance(consistency_checks, dict):
            consistency_checks = {}
        print(f"Experiment artifact consistency: {artifact_consistency.get('status', 'unknown')}")
        print(
            "Artifact consistency coverage: "
            f"matched={consistency_checks.get('matched_values', 0)}/"
            f"{consistency_checks.get('paper_values', 0)}; "
            f"missing={consistency_checks.get('missing_values', 0)}; "
            f"mismatched={consistency_checks.get('mismatched_values', 0)}; "
            f"aggregated={consistency_checks.get('aggregated_values', 0)}; "
            f"csv_artifacts={consistency_checks.get('csv_artifacts', 0)}"
        )
        for error in artifact_consistency.get("errors", []):
            print(f"ARTIFACT CONSISTENCY ERROR: {error}")
        for warning in artifact_consistency.get("warnings", []):
            print(f"ARTIFACT CONSISTENCY WARNING: {warning}")
    if summary_path:
        _write_run_summary_data(summary, summary_path)
        print(f"Validation summary written to {summary_path}")
    return summary


def _experiment_contract_kwargs(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "require_ablation": bool(getattr(args, "require_ablation", True)),
        "require_sensitivity": bool(getattr(args, "require_sensitivity", True)),
        "require_statistical_tests": bool(getattr(args, "require_statistical_tests", True)),
    }


def _experiment_quality_kwargs(
    args: argparse.Namespace,
    *,
    tcga_defaults: bool = False,
) -> dict[str, object]:
    defaults = tcga_experiment_quality_kwargs() if tcga_defaults else {}
    return {
        "expected_datasets": list(getattr(args, "expected_dataset", []) or defaults.get("expected_datasets", [])),
        "expected_metrics": list(getattr(args, "expected_metric", []) or defaults.get("expected_metrics", [])),
        "expected_method": str(getattr(args, "expected_method", "") or defaults.get("expected_method", "")),
        "expected_baseline": str(getattr(args, "expected_baseline", "") or defaults.get("expected_baseline", "")),
    }


def _experiment_provenance_kwargs(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "require_provenance": bool(getattr(args, "require_provenance", False)),
    }


def _experiment_artifact_consistency_kwargs(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "require_artifact_consistency": bool(getattr(args, "require_artifact_consistency", False)),
    }


def _record_result_preflight(state: dict, result_preflight: dict | None) -> None:
    if not result_preflight:
        return
    artifacts = state.setdefault("artifacts", {})
    artifacts["experiment_contract"] = result_preflight.get("experiment_contract", {})
    artifacts["experiment_contract_requirements"] = result_preflight.get(
        "experiment_contract_requirements",
        {},
    )
    artifacts["experiment_quality"] = result_preflight.get("experiment_quality", {})
    artifacts["experiment_provenance"] = result_preflight.get("experiment_provenance", {})
    artifacts["experiment_artifact_consistency"] = result_preflight.get(
        "experiment_artifact_consistency",
        {},
    )


def _validated_results_are_strictly_acceptable(summary: dict) -> bool:
    evidence = summary.get("experiment_evidence", {})
    contract = summary.get("experiment_contract", {})
    return bool(
        evidence.get("real_result_evidence")
        and contract.get("status") == "complete"
        and summary.get("experiment_quality", {}).get("status", "complete") != "invalid"
        and summary.get("experiment_provenance", {}).get("status", "complete") != "invalid"
        and summary.get("experiment_artifact_consistency", {}).get("status", "complete") != "invalid"
    )


def _build_tcga_cohort_summary(dataset_csv_dir: Path) -> str:
    csv_paths = sorted(dataset_csv_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No TCGA cohort CSV files found at {dataset_csv_dir}")

    rows = []
    total_patients = 0
    total_slides = 0
    for csv_path in csv_paths:
        stats = _tcga_csv_stats(csv_path)
        total_patients += stats["patients"]
        total_slides += stats["slides"]
        rows.append(stats)

    lines = [
        "# TCGA Cohort Data Summary for Hyper-ProtoSurv",
        "",
        "This file is generated from the local `dataset_csv/*.csv` files in the "
        "Hyper-ProtoSurv code directory. It contains real TCGA cohort metadata "
        "available in the provided repository: patient identifiers, WSI slide identifiers, "
        "`survival_months`, and `censorship` values.",
        "",
        "This is not a model-performance result file. It does not contain comparative "
        "evaluation metrics, trained-model scores, statistical tests, or component-study results. "
        "The generated paper must therefore describe dataset construction and reserve "
        "performance tables for real experiment outputs.",
        "",
        f"Total unique TCGA patients across cohorts: {total_patients}.",
        f"Total WSI slide rows across cohorts: {total_slides}.",
        "",
        "| Cohort | Unique patients | Slide rows | Censorship=0 | Censorship=1 | Median survival months | Min survival months | Max survival months |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            "| {cohort} | {patients} | {slides} | {censor_0} | {censor_1} | "
            "{median_survival:.2f} | {min_survival:.2f} | {max_survival:.2f} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Completion Notes",
            "",
            "- Add real trained-model performance tables before making comparative performance claims.",
            "- Add reference-method comparison rows only after running the same evaluation protocol.",
            "- Add component-study rows, implementation settings, cross-validation protocol details, "
            "and statistical tests before submission.",
        ]
    )
    return "\n".join(lines)


def _tcga_csv_stats(csv_path: Path) -> dict[str, object]:
    patients = set()
    slides = 0
    censor_0 = 0
    censor_1 = 0
    survival_months: list[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            slides += 1
            case_id = row.get("case_id", "")
            patients.add(_tcga_patient_id(case_id))
            censorship = str(row.get("censorship", "")).strip()
            if censorship == "0":
                censor_0 += 1
            elif censorship == "1":
                censor_1 += 1
            try:
                survival_months.append(float(str(row.get("survival_months", "")).strip()))
            except ValueError:
                continue

    if not survival_months:
        survival_months = [0.0]
    return {
        "cohort": csv_path.stem.upper(),
        "patients": len(patients),
        "slides": slides,
        "censor_0": censor_0,
        "censor_1": censor_1,
        "median_survival": statistics.median(survival_months),
        "min_survival": min(survival_months),
        "max_survival": max(survival_months),
    }


def _tcga_patient_id(case_id: str) -> str:
    parts = str(case_id).split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    return str(case_id).split(".", 1)[0]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_run_summary(state: dict, summary_path: Path, markdown_path: Path | None = None) -> Path:
    return _write_run_summary_data(_build_run_summary(state, markdown_path), summary_path)


def _write_run_reports(
    state: dict,
    *,
    summary_path: Path | None = None,
    markdown_path: Path | None = None,
    acceptance_report_path: Path | None = None,
    default_acceptance_report: bool = True,
    min_llm_sections: int = 4,
    require_llm_self_review: bool = False,
) -> tuple[Path | None, Path | None]:
    resolved_report_path = _resolve_acceptance_report_path(
        acceptance_report_path,
        summary_path,
        markdown_path,
        default_acceptance_report=default_acceptance_report,
    )
    if not summary_path and not resolved_report_path:
        return None, None

    summary = _build_run_summary(state, markdown_path)
    if resolved_report_path:
        summary["outputs"]["acceptance_report_path"] = str(resolved_report_path)

    written_summary_path = _write_run_summary_data(summary, summary_path) if summary_path else None
    written_report_path = (
        _write_acceptance_report(
            summary,
            resolved_report_path,
            min_llm_sections=min_llm_sections,
            require_llm_self_review=require_llm_self_review,
        )
        if resolved_report_path
        else None
    )
    return written_summary_path, written_report_path


def _resolve_acceptance_report_path(
    explicit_path: Path | None,
    summary_path: Path | None,
    markdown_path: Path | None,
    *,
    default_acceptance_report: bool = True,
) -> Path | None:
    if explicit_path:
        return explicit_path
    if not default_acceptance_report:
        return None
    if summary_path:
        return summary_path.with_name("ACCEPTANCE_REPORT.md")
    if markdown_path:
        return markdown_path.with_name("ACCEPTANCE_REPORT.md")
    return None


def _write_run_summary_data(summary: dict, summary_path: Path) -> Path:
    _validate_run_summary_contracts(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def _validate_run_summary_contracts(summary: dict) -> None:
    readiness_contract = summary.get("readiness_contract")
    if readiness_contract is not None:
        _validate_tcga_readiness_contract(readiness_contract)


def _validate_tcga_readiness_contract(contract: object) -> None:
    label = "TCGA readiness_contract"
    if not isinstance(contract, dict):
        raise SystemExit(f"{label} must be a JSON object.")

    required_fields = {
        "schema_version",
        "status",
        "submission_grade",
        "ready_for_submission_grade",
        "ready_for_deterministic_draft",
        "blocking_categories",
        "requirements",
        "next_actions",
    }
    optional_fields = {"pipeline_phase", "artifact_warnings"}
    unknown_fields = sorted(set(contract) - required_fields - optional_fields)
    if unknown_fields:
        raise SystemExit(f"{label} has unknown fields: {', '.join(unknown_fields)}")

    missing_fields = sorted(field for field in required_fields if field not in contract)
    if missing_fields:
        raise SystemExit(f"{label} is missing fields: {', '.join(missing_fields)}")

    if contract["schema_version"] != TCGA_READINESS_CONTRACT_SCHEMA_VERSION:
        raise SystemExit(
            f"{label} schema_version must be {TCGA_READINESS_CONTRACT_SCHEMA_VERSION}."
        )
    if contract["status"] not in {"ready", "blocked"}:
        raise SystemExit(f"{label} status must be ready or blocked.")

    bool_fields = (
        "submission_grade",
        "ready_for_submission_grade",
        "ready_for_deterministic_draft",
    )
    for field in bool_fields:
        if not isinstance(contract[field], bool):
            raise SystemExit(f"{label} {field} must be a boolean.")

    if "pipeline_phase" in contract and not isinstance(contract["pipeline_phase"], str):
        raise SystemExit(f"{label} pipeline_phase must be a string.")

    blocking_categories = contract["blocking_categories"]
    if not isinstance(blocking_categories, list) or not all(
        isinstance(category, str) for category in blocking_categories
    ):
        raise SystemExit(f"{label} blocking_categories must be a string array.")
    unknown_categories = sorted(
        category
        for category in blocking_categories
        if category not in TCGA_READINESS_CONTRACT_CATEGORIES
    )
    if unknown_categories:
        raise SystemExit(
            f"{label} has unknown blocking categories: {', '.join(unknown_categories)}"
        )

    requirements = contract["requirements"]
    if not isinstance(requirements, dict):
        raise SystemExit(f"{label} requirements must be a JSON object.")
    for category, requirement in requirements.items():
        _validate_tcga_readiness_requirement(label, str(category), requirement)

    next_actions = contract["next_actions"]
    if not isinstance(next_actions, list):
        raise SystemExit(f"{label} next_actions must be an array.")
    for index, action in enumerate(next_actions):
        _validate_tcga_readiness_action(label, index, action)

    artifact_warnings = contract.get("artifact_warnings", [])
    if not isinstance(artifact_warnings, list) or not all(
        isinstance(item, str) for item in artifact_warnings
    ):
        raise SystemExit(f"{label} artifact_warnings must be a string array.")


def _validate_tcga_readiness_requirement(label: str, category: str, requirement: object) -> None:
    if not isinstance(requirement, dict):
        raise SystemExit(f"{label} requirements.{category} must be a JSON object.")

    required_fields = {"status", "required", "detail", "next_action", "command"}
    unknown_fields = sorted(set(requirement) - required_fields)
    if unknown_fields:
        raise SystemExit(
            f"{label} requirements.{category} has unknown fields: {', '.join(unknown_fields)}"
        )
    missing_fields = sorted(field for field in required_fields if field not in requirement)
    if missing_fields:
        raise SystemExit(
            f"{label} requirements.{category} is missing fields: {', '.join(missing_fields)}"
        )
    if requirement["status"] not in TCGA_READINESS_REQUIREMENT_STATUSES:
        raise SystemExit(f"{label} requirements.{category}.status is invalid.")
    if not isinstance(requirement["required"], bool):
        raise SystemExit(f"{label} requirements.{category}.required must be a boolean.")
    for field in ("detail", "next_action", "command"):
        if not isinstance(requirement[field], str):
            raise SystemExit(f"{label} requirements.{category}.{field} must be a string.")


def _validate_tcga_readiness_action(label: str, index: int, action: object) -> None:
    if not isinstance(action, dict):
        raise SystemExit(f"{label} next_actions[{index}] must be a JSON object.")

    required_fields = {"category", "action", "command"}
    unknown_fields = sorted(set(action) - required_fields)
    if unknown_fields:
        raise SystemExit(
            f"{label} next_actions[{index}] has unknown fields: {', '.join(unknown_fields)}"
        )
    missing_fields = sorted(field for field in required_fields if field not in action)
    if missing_fields:
        raise SystemExit(
            f"{label} next_actions[{index}] is missing fields: {', '.join(missing_fields)}"
        )
    for field in required_fields:
        if not isinstance(action[field], str):
            raise SystemExit(f"{label} next_actions[{index}].{field} must be a string.")


def _read_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} could not be read as JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return payload


def _write_acceptance_report(
    summary: dict,
    report_path: Path,
    *,
    min_llm_sections: int = 4,
    require_llm_self_review: bool = False,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _build_acceptance_report(
            summary,
            min_llm_sections=min_llm_sections,
            require_llm_self_review=require_llm_self_review,
        ),
        encoding="utf-8",
    )
    return report_path


def _build_acceptance_report(
    summary: dict,
    *,
    min_llm_sections: int = 4,
    require_llm_self_review: bool = False,
) -> str:
    experiment_evidence = _summary_experiment_evidence(summary)
    experiment_contract = _summary_experiment_contract(summary)
    experiment_quality = _summary_experiment_quality(summary)
    experiment_provenance = _summary_experiment_provenance(summary)
    artifact_consistency = _summary_experiment_artifact_consistency(summary)
    checks = _acceptance_checks(
        summary,
        min_llm_sections=min_llm_sections,
        require_llm_self_review=require_llm_self_review,
    )
    failed = [check for check in checks if check["status"] == "FAIL"]
    warnings = [check for check in checks if check["status"] == "WARN"]
    overall = _acceptance_overall(checks)
    pipeline_status = _acceptance_overall(
        [check for check in checks if check["name"] != "Experiment source integrity"]
    )
    submission_evidence_status = _submission_evidence_status(
        experiment_evidence,
        experiment_contract,
        experiment_quality,
        experiment_provenance,
        artifact_consistency,
    )

    inputs = summary.get("inputs", {})
    outputs = summary.get("outputs", {})
    lines = [
        "# Paper Agent Acceptance Report",
        "",
        f"- Overall status: {overall}",
        f"- Pipeline status: {pipeline_status}",
        f"- Submission evidence status: {submission_evidence_status}",
        f"- Project: {summary.get('project_name', '')}",
        f"- Target venue: {summary.get('target_venue', '')}",
        "",
        "## Input Contract",
        "",
        f"- Code path: {inputs.get('code_path', '')}",
        f"- Baseline PDF: {inputs.get('baseline_pdf_path', '')}",
        f"- Experiment results: {inputs.get('experiment_results_path', '') or inputs.get('experiment_results_source', '')}",
        f"- Experiment evidence kind: {experiment_evidence.get('kind', 'unknown')}",
        f"- TCGA artifact-flow summary: {inputs.get('tcga_artifact_flow_summary_path', '') or 'not provided'}",
        f"- TCGA artifact-flow status: {inputs.get('tcga_artifact_flow_summary_status', '') or 'not provided'}",
        f"- Template source: {summary.get('template_source', '')}",
        f"- Network mode: {inputs.get('network_mode', '')}",
        f"- LLM mode: {inputs.get('llm_mode', '')}",
        f"- Submission grade: {inputs.get('submission_grade', False)}",
        (
            f"- LLM provider/model: {inputs.get('llm_provider', '') or 'not recorded'} / "
            f"{inputs.get('llm_model', '') or 'not recorded'}"
        ),
        f"- LLM endpoint host: {inputs.get('llm_endpoint_host', '') or 'not recorded'}",
        (
            f"- LLM preflight: {summary.get('llm_preflight_status', 'not_recorded')}; "
            f"elapsed={summary.get('llm_preflight_elapsed_seconds', 0)}; "
            f"total_tokens={summary.get('llm_preflight_total_tokens', 0)}"
        ),
        (
            f"- LLM section call trace: {summary.get('section_writer_llm_call_successes', 0)}/"
            f"{summary.get('section_writer_llm_call_count', 0)} successful; "
            f"total_tokens={summary.get('section_writer_llm_total_tokens', 0)}"
        ),
        f"- LLM self-review auto revisions: {summary.get('llm_self_review_auto_revisions', 0)}",
        f"- LaTeX compile requested: {inputs.get('latex_compile_requested', False)}",
        "",
        "## Experiment Evidence Coverage",
        "",
        f"- Main result tables: {summary.get('experiment_result_tables', 0)}",
        f"- Ablation evidence items: {summary.get('experiment_ablation_evidence', 0)}",
        f"- Sensitivity evidence items: {summary.get('experiment_sensitivity_evidence', 0)}",
        f"- Statistical test items: {summary.get('experiment_statistical_tests', 0)}",
        f"- Result contract: {experiment_contract.get('status', 'unknown')}",
        f"- Result quality: {experiment_quality.get('status', 'not_configured')}",
        f"- Result provenance: {experiment_provenance.get('status', 'not_configured')}",
        f"- Artifact consistency: {artifact_consistency.get('status', 'not_configured')}",
        "",
        "## Reference Readiness",
        "",
        f"- Resolver mode: {summary.get('reference_resolver_mode', 'not run')}",
        f"- Resolved references: {summary.get('reference_resolved', 0)}",
        f"- Unresolved seed references: {summary.get('reference_unresolved', 0)}",
        f"- Pruned generic seed references: {summary.get('reference_pruned_seed_count', 0)}",
        f"- Related-work candidates: {summary.get('related_work_candidates', 0)}",
        "",
        "## Acceptance Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['name']} | {check['status']} | {check['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Markdown draft: {outputs.get('markdown', '')}",
            f"- LaTeX project: {outputs.get('latex_project_dir', '')}",
            f"- Main TeX: {outputs.get('latex_output_path', '')}",
            f"- Overleaf zip: {outputs.get('latex_zip_path', '')}",
            f"- Draft report: {outputs.get('draft_report_path', '')}",
            f"- Submission checklist: {outputs.get('submission_checklist_path', '')}",
            f"- Figure/table plan: {outputs.get('presentation_plan_path', '')}",
        ]
    )
    if failed:
        lines.extend(
            [
                "",
                "## Blocking Items",
                "",
                *[f"- {check['name']}: {check['detail']}" for check in failed],
            ]
        )
    if warnings:
        lines.extend(
            [
                "",
                "## Warnings",
                "",
                *[f"- {check['name']}: {check['detail']}" for check in warnings],
            ]
        )
    review_details = summary.get("review_finding_details", [])
    if review_details:
        lines.extend(["", "## Reviewer Findings", ""])
        for item in review_details[:10]:
            if not isinstance(item, dict):
                continue
            severity = _table_safe(str(item.get("severity", "")))
            issue = _table_safe(str(item.get("issue", "")))
            suggestion = _table_safe(str(item.get("suggestion", "")))
            lines.append(f"- [{severity}] {issue} Suggestion: {suggestion}")
    return "\n".join(lines) + "\n"


def _acceptance_checks(
    summary: dict,
    *,
    min_llm_sections: int,
    require_llm_self_review: bool,
) -> list[dict[str, str]]:
    inputs = summary.get("inputs", {})
    outputs = summary.get("outputs", {})
    successes = summary.get("section_writer_llm_successes", [])
    attempted = summary.get("section_writer_llm_attempted_sections", [])
    section_errors = summary.get("section_writer_section_errors", {})
    llm_call_count = int(summary.get("section_writer_llm_call_count", 0) or 0)
    llm_call_successes = int(summary.get("section_writer_llm_call_successes", 0) or 0)
    llm_total_tokens = int(summary.get("section_writer_llm_total_tokens", 0) or 0)
    llm_preflight_status = str(summary.get("llm_preflight_status", "not_recorded"))
    llm_preflight_elapsed = summary.get("llm_preflight_elapsed_seconds", 0)
    llm_preflight_tokens = int(summary.get("llm_preflight_total_tokens", 0) or 0)
    compile_status = summary.get("submission_compile_status", "not_run")
    compile_tool = summary.get("submission_compile_tool", "")
    compile_mode = summary.get("submission_compile_mode", "")
    compile_install_hint = summary.get("submission_compile_install_hint", "")
    review_major = int(summary.get("review_findings_major", summary.get("review_findings", 0)) or 0)
    review_minor = int(summary.get("review_findings_minor", 0) or 0)
    experiment_evidence = _summary_experiment_evidence(summary)
    experiment_contract = _summary_experiment_contract(summary)
    experiment_quality = _summary_experiment_quality(summary)
    experiment_provenance = _summary_experiment_provenance(summary)
    artifact_consistency = _summary_experiment_artifact_consistency(summary)
    artifact_flow_item = _tcga_artifact_flow_acceptance_item(inputs)
    checks = [
        _acceptance_item(
            "Input contract",
            bool(inputs.get("code_path") and inputs.get("baseline_pdf_path") and inputs.get("target_venue")),
            (
                f"code={inputs.get('code_path', '')}; baseline={inputs.get('baseline_pdf_path', '')}; "
                f"venue={inputs.get('target_venue', '')}"
            ),
        ),
        _acceptance_item(
            "Experiment input",
            bool(inputs.get("experiment_results_provided")),
            f"source={inputs.get('experiment_results_source', 'none')}; path={inputs.get('experiment_results_path', '')}",
        ),
        _experiment_source_acceptance_item(
            str(experiment_evidence.get("kind", "unknown")),
            str(experiment_evidence.get("note", "")),
        ),
        _acceptance_item(
            "Experiment evidence coverage",
            summary.get("experiment_result_tables", 0) > 0,
            (
                f"main={summary.get('experiment_result_tables', 0)}; "
                f"ablation={summary.get('experiment_ablation_evidence', 0)}; "
                f"sensitivity={summary.get('experiment_sensitivity_evidence', 0)}; "
                f"statistical={summary.get('experiment_statistical_tests', 0)}"
            ),
            warning_status="WARN",
        ),
        _experiment_contract_acceptance_item(experiment_contract),
        *_experiment_quality_acceptance_items(experiment_quality),
        *_experiment_provenance_acceptance_items(experiment_provenance),
        *_experiment_artifact_consistency_acceptance_items(artifact_consistency),
        *([artifact_flow_item] if artifact_flow_item else []),
        *(
            [
                _acceptance_item(
                    "LLM preflight",
                    llm_preflight_status == "pass",
                    (
                        f"status={llm_preflight_status}; "
                        f"elapsed={llm_preflight_elapsed}; "
                        f"total_tokens={llm_preflight_tokens}"
                    ),
                )
            ]
            if llm_preflight_status != "not_recorded"
            else []
        ),
        _acceptance_item(
            "LLM section drafting",
            len(successes) >= min_llm_sections,
            f"{len(successes)}/{len(attempted) or '?'} sections succeeded; required >= {min_llm_sections}; successes={', '.join(successes) or 'none'}",
        ),
        *(
            [
                _acceptance_item(
                    "LLM call trace",
                    llm_call_successes >= min_llm_sections,
                    (
                        f"{llm_call_successes}/{llm_call_count} calls succeeded; "
                        f"total_tokens={llm_total_tokens}; required >= {min_llm_sections}"
                    ),
                )
            ]
            if min_llm_sections > 0 and (llm_call_count or inputs.get("llm_mode") == "required")
            else []
        ),
        _acceptance_item(
            "LLM section errors",
            not section_errors,
            "none" if not section_errors else "; ".join(f"{key}: {value}" for key, value in section_errors.items()),
        ),
        _acceptance_item(
            "Evidence guard",
            summary.get("evidence_guard_findings", 0) == 0,
            f"{summary.get('evidence_guard_findings', 0)} findings",
        ),
        _reviewer_acceptance_item(review_major, review_minor),
        _acceptance_item(
            "Submission readiness",
            summary.get("submission_readiness_status") == "reviewable",
            f"{summary.get('submission_readiness_status', 'not run')} ({summary.get('submission_readiness_score', 0)}/100)",
            warning_status="WARN",
        ),
        _submission_package_acceptance_item(
            str(summary.get("submission_package_status", "not run")),
            int(summary.get("submission_package_errors", 0) or 0),
            int(summary.get("submission_package_warnings", 0) or 0),
        ),
        _compile_acceptance_item(compile_status, compile_tool, compile_mode, str(compile_install_hint)),
        _acceptance_item(
            "Generated figures",
            summary.get("generated_figures", 0) >= min(1, summary.get("presentation_figures", 0)),
            f"{summary.get('generated_figures', 0)}/{summary.get('presentation_figures', 0)} generated",
            warning_status="WARN",
        ),
        _acceptance_item(
            "Output artifacts",
            bool(outputs.get("markdown") and outputs.get("latex_output_path") and outputs.get("draft_report_path")),
            (
                f"markdown={outputs.get('markdown', '')}; "
                f"main_tex={outputs.get('latex_output_path', '')}; "
                f"draft_report={outputs.get('draft_report_path', '')}"
            ),
        ),
    ]
    if require_llm_self_review:
        checks.append(
            _acceptance_item(
                "LLM self-review",
                summary.get("llm_self_review_mode") == "llm",
                f"mode={summary.get('llm_self_review_mode', 'not run')}; unsupported_claims={summary.get('llm_unsupported_claims', 0)}",
            )
        )
    return checks


def _acceptance_overall(checks: list[dict[str, str]]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["status"] == "WARN" for check in checks):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _submission_evidence_status(
    evidence: dict[str, object],
    contract: dict[str, object],
    quality: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
    artifact_consistency: dict[str, object] | None = None,
) -> str:
    kind = str(evidence.get("kind", "unknown"))
    contract_status = str(contract.get("status", "unknown"))
    if kind not in {"real_result_file", "provided_result_text", "structured_state"}:
        return "FAIL"

    evidence_statuses = [
        contract_status,
        str((quality or {}).get("status", "not_configured")),
        str((provenance or {}).get("status", "not_configured")),
        str((artifact_consistency or {}).get("status", "not_configured")),
    ]
    if any(status == "invalid" for status in evidence_statuses):
        return "FAIL"
    if contract_status != "complete":
        return "WARN"
    if str((quality or {}).get("status", "not_configured")) == "needs_attention":
        return "WARN"
    if str((artifact_consistency or {}).get("status", "not_configured")) == "needs_attention":
        return "WARN"
    return "PASS"


def _acceptance_item(
    name: str,
    passed: bool,
    detail: str,
    *,
    warning_status: str = "FAIL",
) -> dict[str, str]:
    return {
        "name": name,
        "status": "PASS" if passed else warning_status,
        "detail": _table_safe(str(detail)),
    }


def _reviewer_acceptance_item(major: int, minor: int) -> dict[str, str]:
    detail = f"{major} major; {minor} minor"
    if major:
        return {"name": "Reviewer", "status": "FAIL", "detail": _table_safe(detail)}
    if minor:
        return {"name": "Reviewer", "status": "WARN", "detail": _table_safe(detail)}
    return {"name": "Reviewer", "status": "PASS", "detail": _table_safe(detail)}


def _submission_package_acceptance_item(status: str, errors: int, warnings: int) -> dict[str, str]:
    detail = f"{status or 'not run'}; errors={errors}; warnings={warnings}"
    if errors or status == "invalid":
        return {"name": "Submission package", "status": "FAIL", "detail": _table_safe(detail)}
    if warnings or status in {"needs_attention", "not run", "not_run", ""}:
        return {"name": "Submission package", "status": "WARN", "detail": _table_safe(detail)}
    return {"name": "Submission package", "status": "PASS", "detail": _table_safe(detail)}


def _experiment_source_acceptance_item(kind: str, note: str) -> dict[str, str]:
    detail = f"kind={kind or 'unknown'}; note={note or 'not recorded'}"
    if kind in {"real_result_file", "provided_result_text", "structured_state"}:
        return {"name": "Experiment source integrity", "status": "PASS", "detail": _table_safe(detail)}
    if kind in {"missing", "synthetic_mock", "data_only", "demo"}:
        return {"name": "Experiment source integrity", "status": "FAIL", "detail": _table_safe(detail)}
    return {"name": "Experiment source integrity", "status": "WARN", "detail": _table_safe(detail)}


def _experiment_contract_acceptance_item(contract: dict[str, object]) -> dict[str, str]:
    checks = contract.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    detail = (
        f"{contract.get('status', 'unknown')}; "
        f"main={checks.get('result_tables', 0)}; "
        f"comparisons={checks.get('numeric_comparisons', 0)}; "
        f"ablation={checks.get('ablation_items', 0)}; "
        f"sensitivity={checks.get('sensitivity_items', 0)}; "
        f"statistical={checks.get('statistical_tests', 0)}"
    )
    status = str(contract.get("status", "unknown"))
    if status == "complete":
        return {"name": "Experiment result contract", "status": "PASS", "detail": _table_safe(detail)}
    if status == "invalid":
        return {"name": "Experiment result contract", "status": "FAIL", "detail": _table_safe(detail)}
    return {"name": "Experiment result contract", "status": "WARN", "detail": _table_safe(detail)}


def _experiment_quality_acceptance_items(quality: dict[str, object]) -> list[dict[str, str]]:
    if quality.get("status") == "not_configured":
        return []
    checks = quality.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    detail = (
        f"{quality.get('status', 'unknown')}; "
        f"missing_datasets={', '.join(checks.get('missing_datasets', []) or []) or 'none'}; "
        f"missing_metrics={', '.join(checks.get('missing_metrics', []) or []) or 'none'}"
    )
    status = str(quality.get("status", "unknown"))
    if status == "complete":
        return [{"name": "Experiment result quality", "status": "PASS", "detail": _table_safe(detail)}]
    if status == "invalid":
        return [{"name": "Experiment result quality", "status": "FAIL", "detail": _table_safe(detail)}]
    return [{"name": "Experiment result quality", "status": "WARN", "detail": _table_safe(detail)}]


def _experiment_provenance_acceptance_items(provenance: dict[str, object]) -> list[dict[str, str]]:
    if provenance.get("status") == "not_configured":
        return []
    checks = provenance.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    detail = (
        f"{provenance.get('status', 'unknown')}; "
        f"entries={checks.get('entries', 0)}; "
        f"local_paths={checks.get('local_paths', 0)}; "
        f"fingerprinted={checks.get('fingerprinted_local_paths', 0)}; "
        f"verified_checksums={checks.get('verified_checksums', 0)}; "
        f"remote_refs={checks.get('remote_references', 0)}; "
        f"missing_paths={checks.get('missing_paths', 0)}"
    )
    status = str(provenance.get("status", "unknown"))
    if status == "complete":
        return [{"name": "Experiment result provenance", "status": "PASS", "detail": _table_safe(detail)}]
    if status == "invalid":
        return [{"name": "Experiment result provenance", "status": "FAIL", "detail": _table_safe(detail)}]
    return [{"name": "Experiment result provenance", "status": "WARN", "detail": _table_safe(detail)}]


def _experiment_artifact_consistency_acceptance_items(consistency: dict[str, object]) -> list[dict[str, str]]:
    if consistency.get("status") == "not_configured":
        return []
    checks = consistency.get("checks", {})
    if not isinstance(checks, dict):
        checks = {}
    detail = (
        f"{consistency.get('status', 'unknown')}; "
        f"matched={checks.get('matched_values', 0)}/{checks.get('paper_values', 0)}; "
        f"missing={checks.get('missing_values', 0)}; "
        f"mismatched={checks.get('mismatched_values', 0)}; "
        f"aggregated={checks.get('aggregated_values', 0)}; "
        f"csv_artifacts={checks.get('csv_artifacts', 0)}"
    )
    status = str(consistency.get("status", "unknown"))
    if status == "complete":
        return [{"name": "Experiment artifact consistency", "status": "PASS", "detail": _table_safe(detail)}]
    if status == "invalid":
        return [{"name": "Experiment artifact consistency", "status": "FAIL", "detail": _table_safe(detail)}]
    return [{"name": "Experiment artifact consistency", "status": "WARN", "detail": _table_safe(detail)}]


def _tcga_artifact_flow_acceptance_item(inputs: dict[str, object]) -> dict[str, str] | None:
    summary_path = str(inputs.get("tcga_artifact_flow_summary_path", "") or "")
    if not summary_path:
        return None
    validation = inputs.get("tcga_artifact_flow_validation", {})
    if not isinstance(validation, dict):
        validation = {}
    status = str(inputs.get("tcga_artifact_flow_summary_status", "unknown") or "unknown")
    phase = str(inputs.get("tcga_artifact_flow_pipeline_phase", "unknown") or "unknown")
    contract_status = str(validation.get("experiment_contract_status", "unknown") or "unknown")
    quality_status = str(validation.get("experiment_quality_status", "unknown") or "unknown")
    provenance_status = str(validation.get("experiment_provenance_status", "unknown") or "unknown")
    consistency_status = str(validation.get("experiment_artifact_consistency_status", "unknown") or "unknown")
    detail = (
        f"summary={summary_path}; status={status}; phase={phase}; "
        f"contract={contract_status}; quality={quality_status}; provenance={provenance_status}; "
        f"consistency={consistency_status}; matched={validation.get('artifact_consistency_matched', 0)}; "
        f"missing={validation.get('artifact_consistency_missing', 0)}; "
        f"mismatched={validation.get('artifact_consistency_mismatched', 0)}"
    )
    validation_statuses = {contract_status, quality_status, provenance_status, consistency_status}
    if status != "pass" or "invalid" in validation_statuses:
        return {"name": "TCGA artifact flow trace", "status": "FAIL", "detail": _table_safe(detail)}
    if "needs_attention" in validation_statuses or "unknown" in validation_statuses:
        return {"name": "TCGA artifact flow trace", "status": "WARN", "detail": _table_safe(detail)}
    return {"name": "TCGA artifact flow trace", "status": "PASS", "detail": _table_safe(detail)}


def _compile_acceptance_item(status: str, tool: str, mode: str, install_hint: str = "") -> dict[str, str]:
    detail = f"status={status}; tool={tool or 'none'}; mode={mode or 'unknown'}"
    if install_hint and status == "tool_unavailable":
        detail += f"; install={install_hint}"
    if status == "passed":
        return _acceptance_item("LaTeX compile", True, detail)
    if status in {"disabled", "tool_unavailable", "not_run", ""}:
        return _acceptance_item("LaTeX compile", False, detail, warning_status="WARN")
    return _acceptance_item("LaTeX compile", False, detail)


def _table_safe(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _summary_experiment_evidence(summary: dict) -> dict[str, object]:
    inputs = summary.get("inputs", {})
    if inputs.get("experiment_evidence_kind"):
        return {
            "kind": inputs.get("experiment_evidence_kind", "unknown"),
            "note": inputs.get("experiment_evidence_note", ""),
        }
    return classify_experiment_evidence(
        source=str(inputs.get("experiment_results_source", "")),
        path=str(inputs.get("experiment_results_path", "")),
        text="",
        result_table_count=int(summary.get("experiment_result_tables", 0) or 0),
    )


def _summary_experiment_contract(summary: dict) -> dict[str, object]:
    contract = summary.get("experiment_contract", {})
    if isinstance(contract, dict) and contract:
        return contract
    checks = {
        "result_tables": int(summary.get("experiment_result_tables", 0) or 0),
        "numeric_comparisons": int(summary.get("experiment_numeric_comparisons", 0) or 0),
        "datasets": int(summary.get("experiment_datasets", 0) or 0),
        "metrics": int(summary.get("experiment_metrics", 0) or 0),
        "ablation_items": int(summary.get("experiment_ablation_evidence", 0) or 0),
        "sensitivity_items": int(summary.get("experiment_sensitivity_evidence", 0) or 0),
        "statistical_tests": int(summary.get("experiment_statistical_tests", 0) or 0),
    }
    errors = []
    warnings = []
    if checks["result_tables"] <= 0:
        errors.append("Missing main trained-model result table with proposed-method and baseline rows.")
    if checks["ablation_items"] <= 0:
        warnings.append("Missing ablation table; component claims should remain provisional.")
    if checks["sensitivity_items"] <= 0:
        warnings.append("Missing sensitivity analysis table; hyperparameter robustness claims should be omitted.")
    if checks["statistical_tests"] <= 0:
        warnings.append("Missing statistical-test table; significance claims should be omitted.")
    return {
        "status": "invalid" if errors else "needs_attention" if warnings else "complete",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def _summary_experiment_quality(summary: dict) -> dict[str, object]:
    quality = summary.get("experiment_quality", {})
    if isinstance(quality, dict) and quality:
        return quality
    return {
        "status": "not_configured",
        "errors": [],
        "warnings": [],
        "checks": {},
    }


def _summary_experiment_provenance(summary: dict) -> dict[str, object]:
    provenance = summary.get("experiment_provenance", {})
    if isinstance(provenance, dict) and provenance:
        return provenance
    return {
        "status": "not_configured",
        "errors": [],
        "warnings": [],
        "entries": [],
        "checks": {},
    }


def _summary_experiment_artifact_consistency(summary: dict) -> dict[str, object]:
    consistency = summary.get("experiment_artifact_consistency", {})
    if isinstance(consistency, dict) and consistency:
        return consistency
    return {
        "status": "not_configured",
        "errors": [],
        "warnings": [],
        "checks": {},
        "matches": [],
        "missing": [],
        "mismatches": [],
    }


def _build_run_summary(state: dict, markdown_path: Path | None = None) -> dict:
    artifacts = state.get("artifacts", {})
    request = state.get("request")
    review_findings = state.get("review_findings", [])
    review_major = sum(1 for finding in review_findings if _review_finding_severity(finding) == "major")
    review_minor = sum(1 for finding in review_findings if _review_finding_severity(finding) == "minor")
    llm_review = artifacts.get("llm_self_review", {})
    reference_verification = artifacts.get("reference_verification", {})
    readiness = artifacts.get("submission_readiness", {})
    code_baseline_comparison = artifacts.get("code_baseline_comparison", {})
    submission_package = artifacts.get("submission_package", {})
    submission_checks = submission_package.get("checks", {}) if isinstance(submission_package, dict) else {}
    compile_check = submission_checks.get("compile", {}) if isinstance(submission_checks, dict) else {}
    presentation_plan = artifacts.get("presentation_plan", {})
    experiments = state.get("experiments")
    experiment_contract = artifacts.get("experiment_contract", {})
    if not isinstance(experiment_contract, dict) or not experiment_contract:
        if experiments:
            experiment_contract = validate_experiment_contract(experiments)
        else:
            experiment_contract = _summary_experiment_contract(
                {
                    "experiment_result_tables": len(artifacts.get("experiment_result_tables", [])),
                    "experiment_ablation_evidence": len(artifacts.get("experiment_ablation_evidence", [])),
                    "experiment_sensitivity_evidence": len(
                        artifacts.get("experiment_sensitivity_evidence", [])
                    ),
                    "experiment_statistical_tests": len(artifacts.get("experiment_statistical_tests", [])),
                }
            )
    experiment_results = getattr(request, "experiment_results", "") or ""
    experiment_quality = artifacts.get("experiment_quality", {})
    if not isinstance(experiment_quality, dict) or not experiment_quality:
        experiment_quality = _summary_experiment_quality({})
    experiment_provenance = artifacts.get("experiment_provenance", {})
    if not isinstance(experiment_provenance, dict) or not experiment_provenance:
        experiment_provenance = _summary_experiment_provenance({})
    artifact_consistency = artifacts.get("experiment_artifact_consistency", {})
    if not isinstance(artifact_consistency, dict) or not artifact_consistency:
        artifact_consistency = _summary_experiment_artifact_consistency({})
    llm_call_trace = _section_writer_llm_call_trace(artifacts)
    llm_preflight = _summary_llm_preflight(artifacts)
    experiment_results_present = bool(experiment_results.strip())
    experiment_results_source = artifacts.get(
        "experiment_results_source", "provided" if experiment_results_present else "none"
    )
    experiment_evidence = classify_experiment_evidence(
        source=str(experiment_results_source),
        path=str(artifacts.get("experiment_results_path", "")),
        text=experiment_results,
        result_table_count=len(artifacts.get("experiment_result_tables", [])),
    )
    return {
        "project_name": getattr(request, "project_name", ""),
        "target_venue": getattr(request, "target_venue", ""),
        "inputs": {
            "code_path": getattr(request, "code_path", "") or "",
            "baseline_pdf_path": getattr(request, "baseline_pdf_path", "") or "",
            "target_venue": getattr(request, "target_venue", ""),
            "experiment_results_provided": experiment_results_present,
            "experiment_results_source": experiment_results_source,
            "experiment_results_path": artifacts.get("experiment_results_path", ""),
            "experiment_evidence_kind": experiment_evidence.get("kind", "unknown"),
            "experiment_evidence_note": experiment_evidence.get("note", ""),
            "tcga_artifact_flow_summary_path": artifacts.get("tcga_artifact_flow_summary_path", ""),
            "tcga_artifact_flow_summary_status": artifacts.get("tcga_artifact_flow_summary_status", ""),
            "tcga_artifact_flow_pipeline_phase": artifacts.get("tcga_artifact_flow_pipeline_phase", ""),
            "tcga_artifact_flow_artifacts_dir": artifacts.get("tcga_artifact_flow_artifacts_dir", ""),
            "tcga_artifact_flow_artifact_files": artifacts.get("tcga_artifact_flow_artifact_files", {}),
            "tcga_artifact_flow_validation": artifacts.get("tcga_artifact_flow_validation", {}),
            "keywords": list(getattr(request, "keywords", []) or []),
            "template_zip_path": getattr(request, "template_zip_path", "") or "",
            "template_dir_path": getattr(request, "template_dir_path", "") or "",
            "network_mode": artifacts.get("runtime_network_mode", "environment"),
            "llm_mode": artifacts.get("runtime_llm_mode", "environment"),
            "llm_provider": artifacts.get("runtime_llm_provider", ""),
            "llm_model": artifacts.get("runtime_llm_model", ""),
            "llm_endpoint_host": artifacts.get("runtime_llm_endpoint_host", ""),
            "llm_configured": bool(artifacts.get("runtime_llm_configured", False)),
            "llm_timeout_seconds": artifacts.get("runtime_llm_timeout_seconds", 0),
            "llm_max_retries": artifacts.get("runtime_llm_max_retries", 0),
            "latex_compile_requested": bool(
                artifacts.get("latex_compile_requested", _truthy_env("PAPER_AGENT_RUN_LATEX_COMPILE"))
            ),
            "min_llm_sections": artifacts.get("min_llm_sections", 0),
            "submission_grade": bool(artifacts.get("submission_grade", False)),
        },
        "section_writer_mode": artifacts.get("section_writer_mode", "unknown"),
        "llm_self_review_mode": llm_review.get("mode", "not run"),
        "llm_unsupported_claims": len(llm_review.get("unsupported_claims", [])),
        "llm_self_review_auto_revisions": len(llm_review.get("auto_revisions", [])),
        "review_findings": len(review_findings),
        "review_findings_major": review_major,
        "review_findings_minor": review_minor,
        "review_finding_details": _review_finding_details(review_findings),
        "submission_readiness_score": readiness.get("overall_score", 0),
        "submission_readiness_status": readiness.get("status", "not run"),
        "submission_package_status": submission_package.get("status", "not run"),
        "submission_package_errors": len(submission_package.get("errors", [])),
        "submission_package_warnings": len(submission_package.get("warnings", [])),
        "submission_compile_mode": compile_check.get("mode", "not_run"),
        "submission_compile_status": compile_check.get("status", "not_run"),
        "submission_compile_tool": compile_check.get("tool", ""),
        "submission_compile_install_hint": compile_check.get("install_hint", ""),
        "presentation_figures": len(presentation_plan.get("figures", [])),
        "generated_figures": len(artifacts.get("generated_figures", [])),
        "presentation_tables": len(presentation_plan.get("tables", [])),
        "presentation_open_items": len(presentation_plan.get("open_items", [])),
        "evidence_guard_findings": len(artifacts.get("evidence_guard_findings", [])),
        "code_baseline_method_shifts": len(
            code_baseline_comparison.get("likely_method_shifts", [])
        ),
        "code_baseline_innovation_seeds": len(
            code_baseline_comparison.get("innovation_seeds", [])
        ),
        "bibliography_entries": len(state.get("bibliography", [])),
        "reference_resolver_mode": artifacts.get("reference_resolver_mode", "not run"),
        "reference_resolved": reference_verification.get("resolved_count", 0),
        "reference_unresolved": reference_verification.get("unresolved_count", 0),
        "reference_pruned_seed_count": len(artifacts.get("reference_pruned_seed_keys", [])),
        "reference_pruned_seed_keys": artifacts.get("reference_pruned_seed_keys", []),
        "reference_resolution_trace": len(artifacts.get("reference_resolution_trace", [])),
        "related_work_candidates": len(artifacts.get("related_work_candidates", [])),
        "experiment_result_tables": len(artifacts.get("experiment_result_tables", [])),
        "experiment_numeric_comparisons": sum(
            len(table.comparisons)
            for table in getattr(experiments, "result_tables", []) or []
        ),
        "experiment_datasets": len(getattr(experiments, "datasets", []) or []),
        "experiment_metrics": len(getattr(experiments, "metrics", []) or []),
        "experiment_contract": experiment_contract,
        "experiment_contract_status": experiment_contract.get("status", "unknown"),
        "experiment_contract_errors": len(experiment_contract.get("errors", [])),
        "experiment_contract_warnings": len(experiment_contract.get("warnings", [])),
        "experiment_quality": experiment_quality,
        "experiment_quality_status": experiment_quality.get("status", "not_configured"),
        "experiment_quality_errors": len(experiment_quality.get("errors", [])),
        "experiment_quality_warnings": len(experiment_quality.get("warnings", [])),
        "experiment_provenance": experiment_provenance,
        "experiment_provenance_status": experiment_provenance.get("status", "not_configured"),
        "experiment_provenance_errors": len(experiment_provenance.get("errors", [])),
        "experiment_provenance_warnings": len(experiment_provenance.get("warnings", [])),
        "experiment_artifact_consistency": artifact_consistency,
        "experiment_artifact_consistency_status": artifact_consistency.get("status", "not_configured"),
        "experiment_artifact_consistency_errors": len(artifact_consistency.get("errors", [])),
        "experiment_artifact_consistency_warnings": len(artifact_consistency.get("warnings", [])),
        "experiment_ablation_evidence": len(artifacts.get("experiment_ablation_evidence", [])),
        "experiment_sensitivity_evidence": len(
            artifacts.get("experiment_sensitivity_evidence", [])
        ),
        "experiment_statistical_tests": len(artifacts.get("experiment_statistical_tests", [])),
        "latex_tables": artifacts.get("latex_table_count", 0),
        "undefined_citation_keys": artifacts.get("undefined_citation_keys", []),
        "template_source": getattr(state.get("venue_template"), "template_source", ""),
        "section_writer_llm_attempted_sections": artifacts.get(
            "section_writer_llm_attempted_sections",
            [],
        ),
        "section_writer_llm_successes": artifacts.get("section_writer_llm_successes", []),
        "section_writer_repaired_sections": artifacts.get("section_writer_repaired_sections", []),
        "section_writer_section_errors": artifacts.get("section_writer_section_errors", {}),
        "section_writer_llm_call_trace": llm_call_trace,
        "section_writer_llm_call_count": len(llm_call_trace),
        "section_writer_llm_call_successes": _section_writer_llm_successful_call_count(llm_call_trace),
        "section_writer_llm_total_tokens": _section_writer_llm_total_tokens(llm_call_trace),
        "llm_preflight": llm_preflight,
        "llm_preflight_status": str(llm_preflight.get("status", "not_recorded")),
        "llm_preflight_elapsed_seconds": llm_preflight.get("elapsed_seconds", 0),
        "llm_preflight_total_tokens": _llm_preflight_total_tokens(llm_preflight),
        "outputs": {
            "markdown": str(markdown_path) if markdown_path else "",
            "latex_project_dir": str(state.get("latex_project_dir", "")),
            "latex_output_path": str(state.get("latex_output_path", "")),
            "latex_zip_path": str(state.get("latex_zip_path", "")),
            "draft_report_path": artifacts.get("draft_report_path", ""),
            "submission_checklist_path": artifacts.get("submission_checklist_path", ""),
            "presentation_plan_path": artifacts.get("presentation_plan_path", ""),
        },
    }


def _summary_llm_preflight(artifacts: dict) -> dict[str, object]:
    value = artifacts.get("paper_e2e_llm_preflight", artifacts.get("llm_preflight", {}))
    return value if isinstance(value, dict) else {}


def _llm_preflight_total_tokens(preflight: dict[str, object]) -> int:
    usage = preflight.get("usage", {})
    if not isinstance(usage, dict):
        return 0
    try:
        return int(usage.get("total_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _section_writer_llm_call_trace(artifacts: dict) -> list[dict[str, object]]:
    trace = artifacts.get("section_writer_llm_call_trace", [])
    if not isinstance(trace, list):
        return []
    return [item for item in trace if isinstance(item, dict)]


def _section_writer_llm_successful_call_count(trace: list[dict[str, object]]) -> int:
    return sum(1 for item in trace if item.get("status") == "success")


def _section_writer_llm_total_tokens(trace: list[dict[str, object]]) -> int:
    total = 0
    for item in trace:
        usage = item.get("usage", {})
        if not isinstance(usage, dict):
            continue
        try:
            total += int(usage.get("total_tokens", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _review_finding_severity(finding: object) -> str:
    if isinstance(finding, dict):
        return str(finding.get("severity", ""))
    return str(getattr(finding, "severity", ""))


def _review_finding_details(findings: list[object], *, limit: int = 20) -> list[dict[str, str]]:
    details = []
    for finding in findings[:limit]:
        if isinstance(finding, dict):
            severity = str(finding.get("severity", ""))
            issue = str(finding.get("issue", ""))
            suggestion = str(finding.get("suggestion", ""))
        else:
            severity = str(getattr(finding, "severity", ""))
            issue = str(getattr(finding, "issue", ""))
            suggestion = str(getattr(finding, "suggestion", ""))
        details.append(
            {
                "severity": severity,
                "issue": issue,
                "suggestion": suggestion,
            }
        )
    return details


def _run_llm_self_review_smoke() -> None:
    config = load_llm_config()
    client = LLMClient(config)
    state = {
        "request": PaperRequest(project_name="llm-self-review-smoke", target_venue="TPAMI"),
        "sections": DraftSections(
            experiments=(
                "We evaluate on BLCA using C-index. The method obtains 0.999 on XYZ."
            )
        ),
        "experiments": ExperimentSummary(
            raw_preview=(
                "| Method | BLCA C-index |\n"
                "|---|---:|\n"
                "| baseline | 0.646 |\n"
                "| ours | 0.671 |\n"
            ),
            datasets=["BLCA"],
            metrics=["C-INDEX"],
            observations=["Ours improves over baseline on BLCA by +0.025."],
        ),
        "innovations": [],
        "bibliography": [],
        "artifacts": {},
    }
    reviewed = LLMSelfReviewAgent(llm_client=client).run(state)
    review = reviewed.get("artifacts", {}).get("llm_self_review", {})
    print(f"LLM self-review mode: {review.get('mode', 'unknown')}")
    if review.get("error"):
        raise SystemExit(f"LLM self-review failed: {review['error']}")
    claims = review.get("unsupported_claims", [])
    print(f"Unsupported claims: {len(claims)}")
    for claim in claims[:3]:
        print(f"- [{claim.get('severity', 'major')}] {claim.get('section')}: {claim.get('claim')}")


if __name__ == "__main__":
    main()
