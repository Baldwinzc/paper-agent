"""Command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_agent.config import load_llm_config
from paper_agent.llm import ChatMessage, LLMClient
from paper_agent.state import PaperRequest
from paper_agent.workflow import PaperWorkflow


def main() -> None:
    parser = argparse.ArgumentParser(prog="paper-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a deterministic demo draft.")
    demo.add_argument("--output", default="outputs/demo", help="Output directory for markdown.")
    sub.add_parser("llm-ping", help="Test the configured OpenAI-compatible LLM.")
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
                "| baseline | 81.2 | 74.5 |\n"
                "| ours | 84.6 | 77.1 |\n"
                "Ablation w/o calibration drops performance."
            ),
            keywords=["representation", "uncertainty", "efficient inference"],
        )
        state = PaperWorkflow().run(request)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "draft.md").write_text(state["final_markdown"], encoding="utf-8")
        print(f"Draft written to {output / 'draft.md'}")
        print(f"LaTeX written to {state['latex_output_path']}")
    elif args.command == "llm-ping":
        config = load_llm_config()
        client = LLMClient(config)
        result = client.chat(
            [
                ChatMessage(role="system", content="You are a concise API health-check assistant."),
                ChatMessage(role="user", content="Reply with exactly: paper-agent-ok"),
            ],
            temperature=0,
            max_tokens=16,
        )
        print(result.content.strip())
