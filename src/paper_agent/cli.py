"""Command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_agent.agents.llm_self_review import LLMSelfReviewAgent
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
    draft.add_argument("--template-zip", default="", help="Optional user-provided LaTeX template zip.")
    draft.add_argument("--template-dir", default="", help="Optional user-provided LaTeX template directory.")
    draft.add_argument(
        "--skip-llm-self-review",
        action="store_true",
        help="Skip the final LLM self-review pass for this run.",
    )
    sub.add_parser("llm-ping", help="Test the configured OpenAI-compatible LLM.")
    sub.add_parser("llm-self-review-smoke", help="Run a tiny configured-LLM self-review smoke test.")
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
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "draft.md").write_text(state["final_markdown"], encoding="utf-8")
        print(f"Draft written to {output / 'draft.md'}")
        print(f"Template source: {state['venue_template'].template_source}")
        print(f"Bibliography entries: {len(state.get('bibliography', []))}")
        print(f"LaTeX tables: {state.get('artifacts', {}).get('latex_table_count', 0)}")
        print(f"LLM self-review: {_llm_self_review_mode(state)}")
        print(f"LaTeX written to {state['latex_output_path']}")
        if args.zip:
            zip_path = zip_latex_project(state["latex_project_dir"], Path(args.zip))
            state["latex_zip_path"] = zip_path
            print(f"Overleaf zip written to {zip_path}")
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
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(state["final_markdown"], encoding="utf-8")
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
            zip_path = zip_latex_project(state["latex_project_dir"], Path(args.zip))
            state["latex_zip_path"] = zip_path
            print(f"Overleaf zip written to {zip_path}")
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
