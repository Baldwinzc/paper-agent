"""Command line interface."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

from paper_agent.agents.draft_report import DraftReportAgent
from paper_agent.agents.llm_self_review import LLMSelfReviewAgent
from paper_agent.agents.submission_package_validator import SubmissionPackageValidatorAgent
from paper_agent.agents.submission_readiness import SubmissionReadinessAgent
from paper_agent.config import load_llm_config
from paper_agent.export import zip_latex_project
from paper_agent.llm import ChatMessage, LLMClient, LLMError
from paper_agent.state import DraftSections, ExperimentSummary, PaperRequest
from paper_agent.workflow import PaperWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(prog="paper-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a deterministic demo draft.")
    demo.add_argument("--output", default="outputs/demo", help="Output directory for markdown.")
    demo.add_argument("--zip", default="", help="Optional path for an Overleaf-ready LaTeX zip.")
    demo.add_argument("--summary", default="", help="Optional path for a JSON run summary.")
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
    draft.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    draft.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    draft.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass for this run.",
    )
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
        "--allow-llm",
        action="store_true",
        help="Allow configured LLM calls. The default sample run is deterministic and local.",
    )
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
    args = parser.parse_args()

    if args.command == "demo":
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
        if args.summary:
            summary_path = _write_run_summary(state, Path(args.summary), markdown_path)
            print(f"Run summary written to {summary_path}")
    elif args.command == "draft":
        baseline_pdf = _resolve_baseline_pdf(args.baseline)
        experiment_results = Path(args.experiment_results).read_text(encoding="utf-8")
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
        state.setdefault("artifacts", {})["experiment_results_source"] = "file"
        state["artifacts"]["experiment_results_path"] = str(Path(args.experiment_results))
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
        print(f"LLM self-review: {_llm_self_review_mode(state)}")
        print(f"LaTeX written to {state['latex_output_path']}")
        if args.zip:
            zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
            print(f"Overleaf zip written to {zip_path}")
        if args.summary:
            summary_path = _write_run_summary(state, Path(args.summary), markdown_path)
            print(f"Run summary written to {summary_path}")
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


def _refresh_submission_artifacts(state: dict) -> None:
    SubmissionPackageValidatorAgent().run(state)
    SubmissionReadinessAgent().run(state)
    DraftReportAgent().run(state)


def _write_latex_zip_and_refresh(state: dict, zip_path: Path) -> Path:
    written_path = zip_latex_project(state["latex_project_dir"], zip_path)
    state["latex_zip_path"] = written_path
    _refresh_submission_artifacts(state)
    return written_path


def _run_hyper_protosurv_sample(args: argparse.Namespace) -> None:
    if not args.allow_llm:
        os.environ["PAPER_AGENT_DISABLE_LLM"] = "1"
    if not args.online:
        os.environ["PAPER_AGENT_DISABLE_TEMPLATE_FETCH"] = "1"
        os.environ["PAPER_AGENT_DISABLE_REFERENCE_RESOLVE"] = "1"
        os.environ["PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY"] = "1"

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
    else:
        dataset_csv_dir = code_path / "dataset_csv"
        experiment_results = _build_tcga_cohort_summary(dataset_csv_dir)
        experiment_results_source = "tcga_cohort_csv"
        experiment_results_path = str(dataset_csv_dir)

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
        skip_llm_self_review=not args.allow_llm,
    )
    state = PaperWorkflow().run(request)
    state.setdefault("artifacts", {})["experiment_results_source"] = experiment_results_source
    state["artifacts"]["experiment_results_path"] = experiment_results_path

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "draft.md"
    markdown_path.write_text(state["final_markdown"], encoding="utf-8")
    print(f"Markdown written to {markdown_path}")

    if args.zip:
        zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
        print(f"Overleaf zip written to {zip_path}")

    summary_path = _write_run_summary(state, output_dir / "RUN_SUMMARY.json", markdown_path)
    print(f"Run summary written to {summary_path}")
    print(f"Review findings: {len(state.get('review_findings', []))}")
    print(f"Template source: {state['venue_template'].template_source}")
    print(f"Bibliography entries: {len(state.get('bibliography', []))}")
    print(f"LLM self-review: {_llm_self_review_mode(state)}")


def _run_llm_draft_smoke(args: argparse.Namespace) -> None:
    config = load_llm_config()
    if not config.configured:
        raise SystemExit(
            "LLM draft smoke requires a configured LLM. Set DEEPSEEK_API_KEY or "
            "OPENAI_API_KEY and TEXT_MODEL, and do not set PAPER_AGENT_DISABLE_LLM=1."
        )
    if not args.online:
        os.environ["PAPER_AGENT_DISABLE_TEMPLATE_FETCH"] = "1"
        os.environ["PAPER_AGENT_DISABLE_REFERENCE_RESOLVE"] = "1"
        os.environ["PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY"] = "1"

    example_root = Path(args.example_root)
    baseline_pdf = _resolve_baseline_pdf(str(example_root / "baseline"))
    code_path = example_root / "code" / "hyper-protosurv"
    if not code_path.is_dir():
        raise SystemExit(f"Hyper-ProtoSurv code directory not found: {code_path}")

    experiment_path = _resolve_project_relative_path(args.experiment_results)
    if not experiment_path.is_file():
        raise SystemExit(f"Experiment results file not found: {experiment_path}")

    request = PaperRequest(
        project_name=args.project_name,
        target_venue=args.target_venue,
        baseline_pdf_path=str(baseline_pdf),
        code_path=str(code_path),
        experiment_results=experiment_path.read_text(encoding="utf-8"),
        keywords=[
            "whole-slide images",
            "survival prediction",
            "computational pathology",
            "hypergraph learning",
        ],
        skip_llm_self_review=not args.include_llm_self_review,
    )
    state = PaperWorkflow(llm_client=LLMClient(config)).run(request)
    state.setdefault("artifacts", {})["experiment_results_source"] = "file"
    state["artifacts"]["experiment_results_path"] = str(experiment_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "draft.md"
    markdown_path.write_text(state["final_markdown"], encoding="utf-8")
    print(f"Markdown written to {markdown_path}")

    if args.zip:
        zip_path = _write_latex_zip_and_refresh(state, Path(args.zip))
        print(f"Overleaf zip written to {zip_path}")

    summary_path = _write_run_summary(state, output_dir / "RUN_SUMMARY.json", markdown_path)
    print(f"Run summary written to {summary_path}")

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
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(_build_run_summary(state, markdown_path), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def _build_run_summary(state: dict, markdown_path: Path | None = None) -> dict:
    artifacts = state.get("artifacts", {})
    request = state.get("request")
    llm_review = artifacts.get("llm_self_review", {})
    reference_verification = artifacts.get("reference_verification", {})
    readiness = artifacts.get("submission_readiness", {})
    code_baseline_comparison = artifacts.get("code_baseline_comparison", {})
    submission_package = artifacts.get("submission_package", {})
    presentation_plan = artifacts.get("presentation_plan", {})
    experiment_results = getattr(request, "experiment_results", "") or ""
    experiment_results_present = bool(experiment_results.strip())
    experiment_results_source = artifacts.get(
        "experiment_results_source", "provided" if experiment_results_present else "none"
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
            "keywords": list(getattr(request, "keywords", []) or []),
            "template_zip_path": getattr(request, "template_zip_path", "") or "",
            "template_dir_path": getattr(request, "template_dir_path", "") or "",
        },
        "section_writer_mode": artifacts.get("section_writer_mode", "unknown"),
        "llm_self_review_mode": llm_review.get("mode", "not run"),
        "llm_unsupported_claims": len(llm_review.get("unsupported_claims", [])),
        "review_findings": len(state.get("review_findings", [])),
        "submission_readiness_score": readiness.get("overall_score", 0),
        "submission_readiness_status": readiness.get("status", "not run"),
        "submission_package_status": submission_package.get("status", "not run"),
        "submission_package_errors": len(submission_package.get("errors", [])),
        "submission_package_warnings": len(submission_package.get("warnings", [])),
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
        "experiment_ablation_evidence": len(artifacts.get("experiment_ablation_evidence", [])),
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
            "presentation_plan_path": artifacts.get("presentation_plan_path", ""),
        },
    }


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
