"""Command line interface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
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
from paper_agent.workflow import PaperWorkflow


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
    _add_experiment_contract_options(tcga_artifacts_doctor)
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
    sub.add_parser("llm-ping", help="Test the configured OpenAI-compatible LLM.")
    llm_doctor = sub.add_parser("llm-doctor", help="Inspect LLM configuration and provider health.")
    llm_doctor.add_argument(
        "--no-live",
        action="store_true",
        help="Only print local configuration; do not call the provider.",
    )
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
    elif args.command == "tcga-artifacts-doctor":
        _run_tcga_artifacts_doctor(args)
    elif args.command == "tcga-preflight":
        _run_tcga_preflight(args)
    elif args.command == "tcga-draft":
        _run_tcga_draft(args)
    elif args.command == "tcga-pipeline":
        _run_tcga_pipeline(args)
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
    elif args.command == "llm-self-review-smoke":
        _run_llm_self_review_smoke()
    elif args.command == "latex-doctor":
        _run_latex_doctor()
    elif args.command == "llm-draft-smoke":
        _run_llm_draft_smoke(args)


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


def _llm_preflight_check(client: LLMClient, config: LLMConfig, *, context: str) -> None:
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


def _llm_preflight_failure_message(config: LLMConfig, context: str, exc: LLMError) -> str:
    raw = _sanitize_llm_error(str(exc), config)
    return (
        f"{context} LLM preflight failed for {_llm_config_label(config)}: "
        f"{_llm_failure_diagnosis(raw)} Raw provider error: {raw}"
    )


def _llm_config_label(config: LLMConfig) -> str:
    parsed = urlparse(config.base_url)
    host = parsed.netloc or parsed.path.split("/")[0]
    return f"{_llm_provider_from_host(host)}/{config.model} at {host or 'unknown-host'}"


def _llm_failure_diagnosis(raw_error: str) -> str:
    lowered = raw_error.lower()
    if "http 402" in lowered or "insufficient balance" in lowered or "insufficient_balance" in lowered:
        return (
            "provider account balance or quota is insufficient. Recharge/enable billing, "
            "switch to another API key, or change provider/model before running generation."
        )
    if "http 401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        return "API authentication failed. Check the configured API key and provider base URL."
    if "http 403" in lowered or "forbidden" in lowered or "permission" in lowered:
        return "the API key is not allowed to use this model or endpoint."
    if "http 404" in lowered or "model_not_found" in lowered or "not found" in lowered:
        return "the configured model or endpoint was not found. Check TEXT_MODEL and API base URL."
    if "timed out" in lowered or "timeout" in lowered:
        return "the provider did not respond before the configured timeout."
    if "transport error" in lowered or "connection" in lowered:
        return "network transport failed before the provider returned a completion."
    return "the provider rejected or failed the health-check request."


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
        return
    if not config.configured:
        print("Live preflight: FAIL")
        raise SystemExit(
            "LLM doctor failed: no configured API key/model. Set DEEPSEEK_API_KEY, "
            "OPENAI_API_KEY, or ARK_API_KEY plus TEXT_MODEL, and ensure PAPER_AGENT_DISABLE_LLM is not enabled."
        )

    client = LLMClient(config)
    try:
        _llm_preflight_check(client, config, context="LLM doctor")
    except SystemExit as exc:
        print("Live preflight: FAIL")
        raise exc
    print("Live preflight: PASS")


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

    print("TCGA project doctor:")
    _print_doctor_check("Example root", example_root.exists(), str(example_root))
    if not example_root.exists():
        blocking.append(f"Example root not found: {example_root}")

    baseline_pdf = None
    try:
        baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
        _print_doctor_check("Baseline PDF", True, str(baseline_pdf))
    except SystemExit as exc:
        _print_doctor_check("Baseline PDF", False, str(example_root / "baseline"))
        blocking.append(str(exc))

    code_path = example_root / "code" / "hyper-protosurv"
    code_ok = code_path.is_dir()
    _print_doctor_check("Code path", code_ok, str(code_path))
    if not code_ok:
        blocking.append(f"Hyper-ProtoSurv code directory not found: {code_path}")

    if not experiment_path.is_file():
        _print_doctor_check("Experiment results", False, str(experiment_path))
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
        _print_doctor_check("Experiment results", True, str(experiment_path))
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
    _print_doctor_check("LLM static config", llm_config.configured, _llm_config_label(llm_config))
    if args.submission_grade and not llm_config.configured:
        blocking.append("Submission-grade TCGA generation requires a configured LLM.")
    if args.live_llm and llm_config.configured:
        try:
            _llm_preflight_check(LLMClient(llm_config), llm_config, context="TCGA doctor")
            _print_doctor_check("LLM live preflight", True, _llm_config_label(llm_config))
        except SystemExit as exc:
            _print_doctor_check("LLM live preflight", False, _llm_config_label(llm_config))
            blocking.append(str(exc))
    elif args.live_llm:
        _print_doctor_check("LLM live preflight", False, "not configured")
        blocking.append("Cannot run LLM live preflight because LLM is not configured.")
    else:
        print("- LLM live preflight: SKIP (pass --live-llm to call the provider)")

    latex_status = _latex_toolchain_status()
    _print_doctor_check(
        "LaTeX compiler",
        bool(latex_status["available"]),
        str(latex_status.get("preferred_tool") or latex_status.get("install_hint")),
    )
    if args.submission_grade and not latex_status["available"]:
        blocking.append(f"LaTeX compiler missing. Install hint: {latex_status['install_hint']}")

    if blocking:
        print("Overall: FAIL")
        print("Blocking items:")
        for item in blocking:
            print(f"- {item}")
        raise SystemExit("TCGA doctor failed: fix blocking items before running tcga-draft.")
    print("Overall: PASS")
    if args.submission_grade:
        print("Ready command: paper-agent tcga-draft --submission-grade")
    else:
        print("Ready command: paper-agent tcga-draft")


def _run_tcga_results_from_artifacts(args: argparse.Namespace) -> None:
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
    _print_doctor_check("Artifact directory", has_input, str(artifact_dir or "not provided"))
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
            _print_doctor_check(label, not required, detail)
            if required:
                blocking.append(f"Missing required {label}: {_tcga_expected_artifact_schema(role)}")
            continue
        diagnostic = _tcga_artifact_role_diagnostic(
            role,
            path,
            args=args,
            datasets=datasets,
        )
        _print_doctor_check(label, diagnostic["ok"], str(diagnostic["detail"]))
        for issue in diagnostic["issues"]:
            print(f"  - {issue}")
        if role == "main" and diagnostic["datasets"]:
            datasets = list(diagnostic["datasets"])
        if not diagnostic["ok"]:
            blocking.append(f"{label} is not parseable: {path}")

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
            else f"{experiment_path} missing and no complete artifact set is available"
        )
        _add_preflight_check(checks, "Experiment results", status, detail, blocking, blocking_item=not artifact_ready)

    llm_config = load_llm_config()
    llm_required = bool(submission_grade or not args.disable_llm)
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
            _add_preflight_check(checks, "LLM live preflight", "FAIL", "LLM is not configured", blocking)
        else:
            try:
                _llm_preflight_check(LLMClient(llm_config), llm_config, context="TCGA preflight")
                _add_preflight_check(checks, "LLM live preflight", "PASS", _llm_config_label(llm_config), blocking, blocking_item=False)
            except SystemExit as exc:
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

    summary = {
        "status": overall.lower(),
        "submission_grade": submission_grade,
        "example_root": str(example_root),
        "experiment_results": str(experiment_path),
        "checks": checks,
        "blocking_items": blocking,
        "result_validation": result_summary,
    }
    if args.summary:
        summary_path = _write_run_summary_data(summary, Path(args.summary))
        print(f"Preflight summary written to {summary_path}")
    if blocking:
        raise SystemExit("TCGA preflight failed: fix blocking items before running tcga-pipeline.")


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


def _print_doctor_check(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"- {name}: {status} ({detail})")


def _run_tcga_pipeline(args: argparse.Namespace) -> None:
    if getattr(args, "submission_grade", False):
        _apply_submission_grade_defaults(args)
    example_root = Path(args.example_root)
    experiment_path = _tcga_result_path(args, example_root)
    args.experiment_results = str(experiment_path)

    if args.skip_result_generation:
        if not experiment_path.is_file():
            raise SystemExit(f"Cannot skip result generation; TCGA result file is missing: {experiment_path}")
        print(f"TCGA pipeline: using existing result file {experiment_path}")
    else:
        print(f"TCGA pipeline: generating result file {experiment_path}")
        result_args = argparse.Namespace(**vars(args))
        result_args.output = str(experiment_path)
        result_args.strict = True
        _run_tcga_results_from_artifacts(result_args)

    if args.skip_doctor:
        print("TCGA pipeline: skipping doctor checks")
    else:
        print("TCGA pipeline: running doctor checks")
        doctor_args = argparse.Namespace(**vars(args))
        doctor_args.write_template = False
        doctor_args.live_llm = bool(args.live_llm_doctor)
        _run_tcga_doctor(doctor_args)

    print("TCGA pipeline: drafting paper")
    _run_tcga_draft(args)


def _run_tcga_draft(args: argparse.Namespace) -> None:
    submission_grade = _apply_submission_grade_defaults(args)
    network_mode = _configure_network_mode(args, default_offline=True)
    compile_latex_requested = _configure_latex_compile(args)

    example_root = Path(args.example_root)
    baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
    code_path = example_root / "code" / "hyper-protosurv"
    if not code_path.is_dir():
        raise SystemExit(f"Hyper-ProtoSurv code directory not found: {code_path}")

    experiment_path = _tcga_result_path(args, example_root)
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
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


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
        f"- Template source: {summary.get('template_source', '')}",
        f"- Network mode: {inputs.get('network_mode', '')}",
        f"- LLM mode: {inputs.get('llm_mode', '')}",
        f"- Submission grade: {inputs.get('submission_grade', False)}",
        (
            f"- LLM provider/model: {inputs.get('llm_provider', '') or 'not recorded'} / "
            f"{inputs.get('llm_model', '') or 'not recorded'}"
        ),
        f"- LLM endpoint host: {inputs.get('llm_endpoint_host', '') or 'not recorded'}",
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
        _acceptance_item(
            "LLM section drafting",
            len(successes) >= min_llm_sections,
            f"{len(successes)}/{len(attempted) or '?'} sections succeeded; required >= {min_llm_sections}; successes={', '.join(successes) or 'none'}",
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
