"""Command line interface."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

from paper_agent.agents.draft_report import DraftReportAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.agents.llm_self_review import LLMSelfReviewAgent
from paper_agent.agents.submission_package_validator import SubmissionPackageValidatorAgent
from paper_agent.agents.submission_readiness import SubmissionReadinessAgent
from paper_agent.config import load_llm_config
from paper_agent.export import zip_latex_project
from paper_agent.experiment_contract import experiment_results_template, validate_experiment_contract
from paper_agent.experiment_evidence import classify_experiment_evidence
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
    sub.add_parser("llm-ping", help="Test the configured OpenAI-compatible LLM.")
    sub.add_parser("llm-self-review-smoke", help="Run a tiny configured-LLM self-review smoke test.")
    llm_draft = sub.add_parser(
        "llm-draft-smoke",
        help="Run a full local draft smoke test and require configured LLM section calls.",
    )
    llm_draft.add_argument("--example-root", default=r"D:\code\agent\example")
    llm_draft.add_argument("--project-name", default="hyper-protosurv-llm-smoke")
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
    args = parser.parse_args()

    if args.command == "validate-results":
        summary = _validate_results_file(
            Path(args.experiment_results),
            summary_path=Path(args.summary) if args.summary else None,
            **_experiment_contract_kwargs(args),
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
        baseline_pdf = _resolve_baseline_pdf(args.baseline)
        experiment_path = _resolve_project_relative_path(args.experiment_results)
        if not experiment_path.is_file():
            raise SystemExit(f"Experiment results file not found: {experiment_path}")
        experiment_results = experiment_path.read_text(encoding="utf-8")
        result_preflight = _validate_results_text(
            experiment_path,
            experiment_results,
            **_experiment_contract_kwargs(args),
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
    elif args.command == "llm-self-review-smoke":
        _run_llm_self_review_smoke()
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


def _record_runtime_modes(
    state: dict,
    *,
    network_mode: str,
    llm_mode: str,
    compile_latex_requested: bool,
    min_llm_sections: int = 0,
) -> None:
    artifacts = state.setdefault("artifacts", {})
    artifacts["runtime_network_mode"] = network_mode
    artifacts["runtime_llm_mode"] = llm_mode
    artifacts["latex_compile_requested"] = compile_latex_requested
    artifacts["min_llm_sections"] = min_llm_sections


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
    )
    if args.strict_results and not _validated_results_are_strictly_acceptable(result_preflight):
        raise SystemExit("Sample failed: experiment result validation failed in strict mode.")

    output_dir = Path(args.output_dir)
    project_name = args.project_name or output_dir.name
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
    )
    if args.strict_results and not _validated_results_are_strictly_acceptable(result_preflight):
        raise SystemExit("LLM draft smoke failed: experiment result validation failed in strict mode.")

    request = PaperRequest(
        project_name=args.project_name,
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
    state = PaperWorkflow(llm_client=LLMClient(config)).run(request)
    _record_runtime_modes(
        state,
        network_mode=network_mode,
        llm_mode="required",
        compile_latex_requested=compile_latex_requested,
        min_llm_sections=args.min_llm_sections,
    )
    state.setdefault("artifacts", {})["experiment_results_source"] = "file"
    state["artifacts"]["experiment_results_path"] = str(experiment_path)
    _record_result_preflight(state, result_preflight)
    SubmissionReadinessAgent().run(state)
    DraftReportAgent().run(state)

    output_dir = Path(args.output_dir)
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


def _validate_results_file(
    path: Path,
    summary_path: Path | None = None,
    *,
    source: str = "file",
    require_ablation: bool = True,
    require_sensitivity: bool = True,
    require_statistical_tests: bool = True,
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
    summary = {
        "path": str(path),
        "source": source,
        "experiment_evidence": evidence,
        "experiment_contract": contract,
        "experiment_contract_requirements": contract.get("requirements", {}),
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


def _record_result_preflight(state: dict, result_preflight: dict | None) -> None:
    if not result_preflight:
        return
    artifacts = state.setdefault("artifacts", {})
    artifacts["experiment_contract"] = result_preflight.get("experiment_contract", {})
    artifacts["experiment_contract_requirements"] = result_preflight.get(
        "experiment_contract_requirements",
        {},
    )


def _validated_results_are_strictly_acceptable(summary: dict) -> bool:
    evidence = summary.get("experiment_evidence", {})
    contract = summary.get("experiment_contract", {})
    return bool(
        evidence.get("real_result_evidence")
        and contract.get("status") == "complete"
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
    checks = _acceptance_checks(
        summary,
        min_llm_sections=min_llm_sections,
        require_llm_self_review=require_llm_self_review,
    )
    failed = [check for check in checks if check["status"] == "FAIL"]
    warnings = [check for check in checks if check["status"] == "WARN"]
    if failed:
        overall = "FAIL"
    elif warnings:
        overall = "PASS_WITH_WARNINGS"
    else:
        overall = "PASS"

    inputs = summary.get("inputs", {})
    outputs = summary.get("outputs", {})
    lines = [
        "# Paper Agent Acceptance Report",
        "",
        f"- Overall status: {overall}",
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
        f"- LaTeX compile requested: {inputs.get('latex_compile_requested', False)}",
        "",
        "## Experiment Evidence Coverage",
        "",
        f"- Main result tables: {summary.get('experiment_result_tables', 0)}",
        f"- Ablation evidence items: {summary.get('experiment_ablation_evidence', 0)}",
        f"- Sensitivity evidence items: {summary.get('experiment_sensitivity_evidence', 0)}",
        f"- Statistical test items: {summary.get('experiment_statistical_tests', 0)}",
        f"- Result contract: {experiment_contract.get('status', 'unknown')}",
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
    review_major = int(summary.get("review_findings_major", summary.get("review_findings", 0)) or 0)
    review_minor = int(summary.get("review_findings_minor", 0) or 0)
    experiment_evidence = _summary_experiment_evidence(summary)
    experiment_contract = _summary_experiment_contract(summary)
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
        _compile_acceptance_item(compile_status, compile_tool, compile_mode),
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
    if kind == "missing":
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


def _compile_acceptance_item(status: str, tool: str, mode: str) -> dict[str, str]:
    detail = f"status={status}; tool={tool or 'none'}; mode={mode or 'unknown'}"
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
            "latex_compile_requested": bool(
                artifacts.get("latex_compile_requested", _truthy_env("PAPER_AGENT_RUN_LATEX_COMPILE"))
            ),
            "min_llm_sections": artifacts.get("min_llm_sections", 0),
        },
        "section_writer_mode": artifacts.get("section_writer_mode", "unknown"),
        "llm_self_review_mode": llm_review.get("mode", "not run"),
        "llm_unsupported_claims": len(llm_review.get("unsupported_claims", [])),
        "review_findings": len(review_findings),
        "review_findings_major": review_major,
        "review_findings_minor": review_minor,
        "submission_readiness_score": readiness.get("overall_score", 0),
        "submission_readiness_status": readiness.get("status", "not run"),
        "submission_package_status": submission_package.get("status", "not run"),
        "submission_package_errors": len(submission_package.get("errors", [])),
        "submission_package_warnings": len(submission_package.get("warnings", [])),
        "submission_compile_mode": compile_check.get("mode", "not_run"),
        "submission_compile_status": compile_check.get("status", "not_run"),
        "submission_compile_tool": compile_check.get("tool", ""),
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
