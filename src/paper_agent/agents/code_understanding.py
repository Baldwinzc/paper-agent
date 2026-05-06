"""Lightweight codebase understanding."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from paper_agent.state import CodeSummary, PaperRequest, PaperState


class CodeUnderstandingAgent:
    """Scans our code as evidence for method modules."""

    METHOD_HINTS = ("model", "module", "loss", "train", "trainer", "network", "attention", "encoder")
    TEXT_HINTS = (
        "replace",
        "replaces",
        "removed",
        "propose",
        "introduce",
        "adaptive",
        "hyperedge",
        "prototype",
        "loss",
        "fusion",
        "ablation",
        "objective",
    )

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        root = Path(request.code_path) if request.code_path else None
        if not root or not root.exists():
            state["code"] = CodeSummary(summary="No code path provided yet.")
            return state

        files = [
            p
            for p in root.rglob("*")
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        ]
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
        method_claims = self._extract_method_claims(root, files)

        state["code"] = CodeSummary(
            path=str(root),
            languages=dict(languages.most_common(10)),
            likely_entrypoints=entrypoints,
            likely_method_files=method_files,
            method_claims=method_claims,
            summary=(
                f"Scanned {len(files)} files. Likely method-bearing files: "
                f"{', '.join(method_files[:5]) or 'not identified'}. "
                f"Extracted {len(method_claims)} method claims from repository text."
            ),
        )
        return state

    def _extract_method_claims(self, root: Path, files: list[Path]) -> list[str]:
        priority_names = {"readme.md", "protosurv.yml", "config.yaml", "config.yml"}
        candidates = [
            p
            for p in files
            if p.name.lower() in priority_names
            or ("model" in p.name.lower() and p.suffix.lower() == ".py")
            or ("loss" in p.name.lower() and p.suffix.lower() == ".py")
        ][:12]
        claims: list[str] = []
        for path in candidates:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for passage in self._candidate_passages(text):
                claim = self._clean_claim(passage)
                if len(claim) < 30 or len(claim) > 420:
                    continue
                lowered = claim.lower()
                if any(hint in lowered for hint in self.TEXT_HINTS):
                    rel = path.relative_to(root)
                    claims.append(f"{claim} [{rel}]")
                if len(claims) >= 12:
                    return list(dict.fromkeys(claims))
        return list(dict.fromkeys(claims))

    def _candidate_passages(self, text: str) -> list[str]:
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        paragraphs = [
            " ".join(block.split())
            for block in re.split(r"\n\s*\n", text)
            if block.strip()
        ]
        passages: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= 360:
                passages.append(paragraph)
                continue
            passages.extend(
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?。])\s+", paragraph)
                if sentence.strip()
            )
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return passages + lines

    def _clean_claim(self, text: str) -> str:
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"[_#>*|]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" -:\t")
        lowered = text.lower()
        if re.match(r"^\d+(\.\d+)*\.?\s+", text):
            return ""
        if any(
            token in lowered
            for token in [
                "bash ",
                "python scripts/",
                "data root=",
                "slots=",
                "stdout/stderr",
                "run order",
                "dispatcher",
                "data root must",
                "the four ablations correspond",
                "original protosurv term",
            ]
        ):
            return ""
        if lowered in {"the training objective is intentionally minimal"}:
            return ""
        if text.startswith(("cp ", "bash ", "python ", "kill ", "nvidia-smi")):
            return ""
        return text
