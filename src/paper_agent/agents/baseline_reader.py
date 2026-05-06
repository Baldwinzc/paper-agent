"""Baseline paper reader."""

from __future__ import annotations

from pathlib import Path

from paper_agent.state import BaselineSummary, PaperRequest, PaperState


class BaselineReaderAgent:
    """Extracts a coarse baseline-paper summary from a PDF or text fallback."""

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        text = self._extract_text(request.baseline_pdf_path)
        preview = self._compact(text)
        limitations = self._guess_limitations(text, request.method_notes)
        terms = self._guess_terms(text, request.keywords)

        state["baseline"] = BaselineSummary(
            title=self._guess_title(text) or "Baseline Paper",
            problem=self._guess_sentence(text, ["problem", "challenge", "task"]) or "To be refined.",
            method=self._guess_sentence(text, ["method", "framework", "model"]) or "To be refined.",
            experiments=self._guess_sentence(text, ["experiment", "dataset", "evaluation"])
            or "To be refined.",
            limitations=limitations,
            related_terms=terms,
            extracted_text_preview=preview,
        )
        return state

    def _extract_text(self, pdf_path: str | None) -> str:
        if not pdf_path:
            return ""
        path = Path(pdf_path)
        if not path.exists():
            return ""
        if path.suffix.lower() != ".pdf":
            return path.read_text(encoding="utf-8", errors="ignore")

        try:
            import fitz  # type: ignore

            doc = fitz.open(path)
            return "\n".join(page.get_text("text") for page in doc[:8])
        except Exception:
            return ""

    def _compact(self, text: str, limit: int = 2500) -> str:
        return " ".join(text.split())[:limit]

    def _guess_title(self, text: str) -> str:
        for line in text.splitlines():
            clean = line.strip()
            if 8 <= len(clean) <= 160 and not clean.lower().startswith(("abstract", "introduction")):
                return clean
        return ""

    def _guess_sentence(self, text: str, keywords: list[str]) -> str:
        for sentence in text.replace("\n", " ").split("."):
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords) and len(sentence.strip()) > 40:
                return sentence.strip() + "."
        return ""

    def _guess_limitations(self, text: str, method_notes: str) -> list[str]:
        candidates = []
        haystack = f"{text}\n{method_notes}".lower()
        if "efficien" in haystack or "cost" in haystack:
            candidates.append("Efficiency or computational cost appears to be a possible limitation.")
        if "robust" in haystack or "noise" in haystack:
            candidates.append("Robustness under noisy or shifted conditions appears relevant.")
        if "general" in haystack or "transfer" in haystack:
            candidates.append("Generalization across datasets or settings appears relevant.")
        return candidates or ["The baseline limitations need explicit user confirmation."]

    def _guess_terms(self, text: str, keywords: list[str]) -> list[str]:
        terms = list(dict.fromkeys(keywords))
        for token in ["representation", "attention", "optimization", "adaptation", "retrieval"]:
            if token in text.lower() and token not in terms:
                terms.append(token)
        return terms[:8]

