"""Lightweight codebase understanding."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from paper_agent.state import CodeSummary, PaperRequest, PaperState


class CodeUnderstandingAgent:
    """Scans our code as evidence for method modules."""

    METHOD_HINTS = ("model", "module", "loss", "train", "trainer", "network", "attention", "encoder")

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        root = Path(request.code_path) if request.code_path else None
        if not root or not root.exists():
            state["code"] = CodeSummary(summary="No code path provided yet.")
            return state

        files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
        languages = Counter(p.suffix.lower() or "<none>" for p in files)
        method_files = [
            str(p.relative_to(root))
            for p in files
            if any(hint in p.name.lower() for hint in self.METHOD_HINTS)
        ][:20]
        entrypoints = [
            str(p.relative_to(root))
            for p in files
            if p.name.lower() in {"main.py", "train.py", "run.py", "config.yaml", "config.yml"}
        ][:20]

        state["code"] = CodeSummary(
            path=str(root),
            languages=dict(languages.most_common(10)),
            likely_entrypoints=entrypoints,
            likely_method_files=method_files,
            summary=(
                f"Scanned {len(files)} files. Likely method-bearing files: "
                f"{', '.join(method_files[:5]) or 'not identified'}."
            ),
        )
        return state

