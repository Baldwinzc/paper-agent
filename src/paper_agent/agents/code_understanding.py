"""Lightweight codebase understanding."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from paper_agent.state import CodeSummary, PaperRequest, PaperState


class CodeUnderstandingAgent:
    """Scans our code as evidence for method modules."""

    METHOD_HINTS = ("model", "module", "loss", "train", "trainer", "network", "attention", "encoder")
    IMPLEMENTATION_PATTERNS = (
        ("OT/Wasserstein hypergraph construction", r"free_support_barycenter|ot\.emd|compute_cost_matrix|cross-cluster|M_OT"),
        ("prototype source", r"data\.prototypes|proto_query_source|prototype-query|prototype bank"),
        ("BHE/HCoN module", r"\bHCoN\b|self\.hcon|ablate_bidirectional|hyperedge-side"),
        ("cross-attention fusion", r"proto_fusion|CrossAttention|cross-attention|prototypes_q"),
        ("reconstruction objective", r"L_rec|hcon_rec_loss|binary_cross_entropy|lambda_rec|hcon_beta"),
        ("survival objective", r"loss_surv|L_surv|Cox PH|partial likelihood"),
        ("legacy loss removal", r"compatibility loss|orthogonal_regularization|online prototype bank|L_comp|L_ortho"),
    )
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
        implementation_evidence = self._extract_implementation_evidence(root, files)

        state["code"] = CodeSummary(
            path=str(root),
            languages=dict(languages.most_common(10)),
            likely_entrypoints=entrypoints,
            likely_method_files=method_files,
            implementation_evidence=implementation_evidence,
            method_claims=method_claims,
            summary=(
                f"Scanned {len(files)} files. Likely method-bearing files: "
                f"{', '.join(method_files[:5]) or 'not identified'}. "
                f"Extracted {len(method_claims)} method claims and "
                f"{len(implementation_evidence)} implementation evidence snippets from repository text."
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

    def _extract_implementation_evidence(self, root: Path, files: list[Path]) -> list[str]:
        candidates = self._implementation_candidate_files(root, files)
        evidence: list[str] = []
        per_file_counts: Counter[Path] = Counter()
        seen_signatures: set[str] = set()
        seen_labels: set[str] = set()
        for path in candidates:
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            path_matches: list[tuple[int, int, str, str, str]] = []
            for lineno, raw_line in enumerate(lines, start=1):
                cleaned = self._clean_evidence_line(raw_line)
                if not cleaned:
                    continue
                label = self._implementation_label(cleaned)
                if not label:
                    continue
                signature = re.sub(r"\W+", " ", cleaned.lower()).strip()
                priority = self._evidence_line_priority(raw_line, cleaned)
                path_matches.append((priority, lineno, label, cleaned, signature))
            selected_matches = self._select_file_evidence(path_matches)
            for _, lineno, label, cleaned, signature in selected_matches:
                if signature in seen_signatures or per_file_counts[path] >= 6:
                    continue
                rel = path.relative_to(root)
                evidence.append(f"{rel}:{lineno} ({label}) {cleaned}")
                per_file_counts[path] += 1
                seen_signatures.add(signature)
                seen_labels.add(label)
                if len(evidence) >= 24 and len(seen_labels) >= 6:
                    return evidence
        return evidence[:24]

    def _select_file_evidence(
        self,
        path_matches: list[tuple[int, int, str, str, str]],
    ) -> list[tuple[int, int, str, str, str]]:
        sorted_matches = sorted(path_matches)
        selected: list[tuple[int, int, str, str, str]] = []
        labels: set[str] = set()
        for item in sorted_matches:
            label = item[2]
            if label in labels:
                continue
            selected.append(item)
            labels.add(label)
            if len(selected) >= 6:
                return selected
        for item in sorted_matches:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= 6:
                break
        return selected

    def _implementation_candidate_files(self, root: Path, files: list[Path]) -> list[Path]:
        def score(path: Path) -> tuple[int, str]:
            rel = str(path.relative_to(root)).lower().replace("\\", "/")
            value = 100
            if path.suffix.lower() not in {".py", ".yml", ".yaml", ".md"}:
                value += 100
            if "readme" in path.name.lower():
                value += 20
            if "config" in rel:
                value -= 35
            if "model" in rel:
                value -= 32
            if "loss" in rel or "core_funcs" in rel or "train.py" in rel:
                value -= 30
            if "data_preparation" in rel or "hypergraph" in rel:
                value -= 36
            if "__pycache__" in rel:
                value += 100
            return (value, rel)

        relevant = []
        for path in files:
            lowered = path.name.lower()
            rel = str(path.relative_to(root)).lower().replace("\\", "/")
            if path.suffix.lower() not in {".py", ".yml", ".yaml", ".md"}:
                continue
            if "__pycache__" in path.parts:
                continue
            if (
                lowered in {"readme.md", "config.yaml", "config.yml", "protosurv.yml"}
                or any(hint in lowered for hint in self.METHOD_HINTS)
                or "core_funcs" in rel
                or "loss" in rel
                or "train.py" in rel
                or "data_preparation" in rel
                or "hypergraph" in rel
            ):
                relevant.append(path)
        return sorted(relevant, key=score)[:40]

    def _clean_evidence_line(self, line: str) -> str:
        text = line.strip()
        if not text:
            return ""
        text = re.sub(r"^\s*(#|//|--)\s*", "", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip(" -:\t")
        if len(text) < 18 or len(text) > 220:
            return ""
        lowered = text.lower()
        if text.startswith(("import ", "from ", "print(", "parser =", "logger.")):
            return ""
        if any(token in lowered for token in ["todo", "copyright", "license", "usage:"]):
            return ""
        return text

    def _implementation_label(self, cleaned_line: str) -> str:
        for label, pattern in self.IMPLEMENTATION_PATTERNS:
            if re.search(pattern, cleaned_line, flags=re.I):
                return label
        return ""

    def _evidence_line_priority(self, raw_line: str, cleaned_line: str) -> int:
        priority = 50
        stripped = raw_line.strip()
        if any(
            token in cleaned_line
            for token in [
                "self.hcon",
                "binary_cross_entropy",
                "free_support_barycenter",
                "loss =",
                "data.prototypes",
                "parser.add_argument",
            ]
        ):
            priority -= 25
        if re.search(r"\b(class|def)\s+\w+", cleaned_line):
            priority -= 18
        if "=" in cleaned_line or "(" in cleaned_line:
            priority -= 10
        if stripped.startswith(("#", "//")):
            priority += 8
        if stripped.startswith(('"""', "'''", "*", ">")):
            priority += 10
        if cleaned_line.lower().startswith(("described in the paper", "where ", "images that ")):
            priority += 15
        return priority
