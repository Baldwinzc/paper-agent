"""Baseline paper reader."""

from __future__ import annotations

import re
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
        path_title = self._guess_title_from_path(request.baseline_pdf_path)
        text_title = self._guess_title(text)

        state["baseline"] = BaselineSummary(
            title=self._best_title(text_title, path_title) or "Baseline Paper",
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
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines[:12]):
            clean = line.strip()
            lowered = clean.lower()
            if lowered.startswith(("abstract", "introduction", "proceedings", "published as")):
                continue
            if 8 <= len(clean) <= 180 and not self._looks_like_author_line(clean):
                return clean
            if index + 1 < len(lines):
                combined = f"{clean} {lines[index + 1].strip()}"
                if 20 <= len(combined) <= 220 and not self._looks_like_author_line(combined):
                    return combined
        return ""

    def _guess_title_from_path(self, pdf_path: str | None) -> str:
        if not pdf_path:
            return ""
        stem = Path(pdf_path).stem
        stem = re.sub(r"^(neurips|nips|icml|iclr|cvpr|iccv|eccv|acl|emnlp)[-_ ]+\d{4}[-_ ]+", "", stem, flags=re.I)
        stem = re.sub(r"[-_ ]+(paper[-_ ]+conference|conference[-_ ]+paper)$", "", stem, flags=re.I)
        stem = re.sub(r"[-_ ]+(paper|conference|main|camera[-_ ]?ready)$", "", stem, flags=re.I)
        stem = re.sub(r"[-_ ]+", " ", stem).strip()
        if len(stem) < 20:
            return ""
        return self._title_case(stem)

    def _best_title(self, text_title: str, path_title: str) -> str:
        if not text_title:
            return path_title
        if not path_title:
            return text_title
        text_tokens = self._title_tokens(text_title)
        path_tokens = self._title_tokens(path_title)
        if len(path_tokens) >= len(text_tokens) + 3 and text_tokens <= path_tokens:
            return path_title
        if len(text_tokens) < 5 <= len(path_tokens):
            return path_title
        return text_title

    def _looks_like_author_line(self, text: str) -> bool:
        lowered = text.lower()
        if "@" in text or "university" in lowered or "institute" in lowered:
            return True
        return text.count(",") >= 2 and not any(
            keyword in lowered for keyword in ["learning", "prediction", "representation", "survival"]
        )

    def _title_case(self, text: str) -> str:
        small_words = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
        words = []
        for index, word in enumerate(text.split()):
            lowered = word.lower()
            if index > 0 and lowered in small_words:
                words.append(lowered)
            elif word.isupper():
                words.append(word)
            else:
                words.append(word[:1].upper() + word[1:])
        return " ".join(words)

    def _title_tokens(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2}

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
