"""Baseline paper reader."""

from __future__ import annotations

import re
from pathlib import Path

from paper_agent.state import BaselineSummary, PaperRequest, PaperState


class BaselineReaderAgent:
    """Extracts a coarse baseline-paper summary from a PDF or text fallback."""

    SECTION_ALIASES = {
        "abstract": {"abstract", "summary"},
        "introduction": {"introduction"},
        "related_work": {
            "related work",
            "background",
            "literature review",
        },
        "method": {
            "method",
            "methods",
            "methodology",
            "approach",
            "proposed method",
            "model",
            "framework",
            "preliminaries",
        },
        "experiments": {
            "experiment",
            "experiments",
            "experimental setup",
            "experimental results",
            "results",
            "evaluation",
        },
        "conclusion": {
            "conclusion",
            "conclusions",
            "discussion",
            "limitations",
            "future work",
        },
    }
    SECTION_ORDER = ("abstract", "introduction", "related_work", "method", "experiments", "conclusion")

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        raw_text = self._extract_text(request.baseline_pdf_path)
        reference_text = self._extract_text(request.baseline_pdf_path, max_pages=None)
        text = self._clean_extracted_text(raw_text)
        structured_sections = self._extract_structured_sections(raw_text)
        section_text = self._merged_section_text(structured_sections, text)
        preview = self._compact(text)
        limitations = self._guess_limitations(section_text, request.method_notes)
        terms = self._guess_terms(section_text, request.keywords)
        path_title = self._guess_title_from_path(request.baseline_pdf_path)
        text_title = self._guess_title(raw_text)

        state["baseline"] = BaselineSummary(
            title=self._best_title(text_title, path_title) or "Baseline Paper",
            problem=self._baseline_problem(structured_sections, text),
            method=self._baseline_method(structured_sections, text),
            experiments=self._baseline_experiments(structured_sections, text),
            limitations=limitations,
            related_terms=terms,
            structured_sections=structured_sections,
            references=self._extract_references(reference_text or raw_text),
            extracted_text_preview=preview,
        )
        return state

    def _extract_text(self, pdf_path: str | None, max_pages: int | None = 8) -> str:
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
            pages = doc if max_pages is None else doc[:max_pages]
            return "\n".join(page.get_text("text") for page in pages)
        except Exception:
            return ""

    def _extract_references(self, text: str) -> dict[str, str]:
        if not text:
            return {}
        match = re.search(r"\bReferences\b", text, flags=re.I)
        if not match:
            return {}
        body = text[match.end() :]
        body = re.sub(r"(?<=\w)-\s+(?=\w)", "", body)
        body = re.split(r"\n[A-Z]\n(?:Appendix|Additional|Implementation|Dataset|Visualization)", body, maxsplit=1)[0]
        lines = []
        for raw in body.splitlines():
            clean = " ".join(raw.split()).strip()
            if not clean or re.fullmatch(r"\d+", clean):
                continue
            lines.append(clean)
        joined = "\n".join(lines)
        references: dict[str, str] = {}
        for chunk in re.split(r"\n(?=\[\d+\]\s+)", joined):
            ref_match = re.match(r"\[(\d+)\]\s+(.+)", chunk, flags=re.S)
            if not ref_match:
                continue
            label = ref_match.group(1)
            reference = " ".join(ref_match.group(2).split())
            if 25 <= len(reference) <= 1200:
                references[label] = reference[:800]
        return references

    def _clean_extracted_text(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            clean = " ".join(line.split()).strip()
            if not clean or self._is_pdf_noise_line(clean):
                continue
            clean = self._strip_section_prefix(clean)
            if clean:
                lines.append(clean)
        return "\n".join(lines)

    def _strip_section_prefix(self, text: str) -> str:
        pattern = re.compile(
            r"^(?:[a-z]{2}\s+)?(?:abstract|summary|introduction|keywords|index terms)"
            r"\b[:.\-\s]*",
            flags=re.I,
        )
        stripped = pattern.sub("", text, count=1).strip()
        if stripped == text:
            return text
        return stripped

    def _is_pdf_noise_line(self, text: str) -> bool:
        lowered = text.lower()
        if "@" in lowered:
            return True
        return lowered in {"cn", "com", "edu", "org"} or bool(re.fullmatch(r"\d+", lowered))

    def _compact(self, text: str, limit: int = 2500) -> str:
        text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
        return " ".join(text.split())[:limit]

    def _extract_structured_sections(self, raw_text: str) -> dict[str, str]:
        lines = self._normalized_lines(raw_text)
        sections: dict[str, list[str]] = {}
        current = ""
        index = 0
        while index < len(lines):
            line = lines[index]
            heading = self._section_key(line)
            if not heading and re.fullmatch(r"\d+(?:\.\d+)*", line) and index + 1 < len(lines):
                heading = self._section_key(lines[index + 1])
                if heading:
                    index += 1
            if heading:
                current = heading
                sections.setdefault(current, [])
                index += 1
                continue
            if current:
                sections.setdefault(current, []).append(line)
            index += 1

        compacted = {
            key: self._compact(" ".join(values), limit=1800)
            for key, values in sections.items()
            if values
        }
        return {key: compacted[key] for key in self.SECTION_ORDER if key in compacted}

    def _normalized_lines(self, text: str) -> list[str]:
        lines = []
        for raw in text.splitlines():
            clean = " ".join(raw.split()).strip()
            if not clean or self._is_pdf_noise_line(clean):
                continue
            if self._looks_like_author_line(clean) or self._looks_like_affiliation_line(clean):
                continue
            lines.append(clean)
        return lines

    def _section_key(self, line: str) -> str:
        normalized = self._normalize_heading(line)
        for section, aliases in self.SECTION_ALIASES.items():
            if normalized in aliases:
                return section
        return ""

    def _normalize_heading(self, line: str) -> str:
        heading = line.strip()
        heading = re.sub(r"^\d+(?:\.\d+)*\s*", "", heading)
        heading = re.sub(r"^[ivxlcdm]+\.\s*", "", heading, flags=re.I)
        heading = re.sub(r"[^A-Za-z ]+", " ", heading)
        heading = re.sub(r"\s+", " ", heading).strip().lower()
        return heading

    def _looks_like_affiliation_line(self, text: str) -> bool:
        lowered = text.lower()
        affiliation_markers = [
            "university",
            "institute",
            "laboratory",
            "school of",
            "department of",
            "academy of",
            "ministry of",
            "equal contribution",
            "corresponding author",
            "conference on",
        ]
        return any(marker in lowered for marker in affiliation_markers)

    def _merged_section_text(self, sections: dict[str, str], fallback_text: str) -> str:
        if not sections:
            return fallback_text
        return "\n".join(sections.get(key, "") for key in self.SECTION_ORDER if sections.get(key))

    def _baseline_problem(self, sections: dict[str, str], text: str) -> str:
        source = " ".join(
            part for part in [sections.get("abstract", ""), sections.get("introduction", "")] if part
        ) or text
        return self._best_keyword_sentence(
            source,
            ["challenge", "problem", "task", "however", "neglect", "limitation"],
        ) or "To be refined."

    def _baseline_method(self, sections: dict[str, str], text: str) -> str:
        source = " ".join(
            part
            for part in [
                sections.get("abstract", ""),
                sections.get("introduction", ""),
                sections.get("method", ""),
            ]
            if part
        ) or text
        return self._best_keyword_sentence(
            source,
            [
                "propose",
                "proposed",
                "method",
                "framework",
                "model",
                "prototype",
                "heterogeneous graph",
            ],
        ) or "To be refined."

    def _baseline_experiments(self, sections: dict[str, str], text: str) -> str:
        source = " ".join(
            part
            for part in [
                sections.get("abstract", ""),
                sections.get("introduction", ""),
                sections.get("experiments", ""),
            ]
            if part
        ) or text
        return self._best_keyword_sentence(
            source,
            [
                "validate",
                "evaluated",
                "evaluation",
                "experiment",
                "dataset",
                "tcga",
                "benchmark",
            ],
        ) or "To be refined."

    def _guess_title(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index, line in enumerate(lines[:12]):
            clean = line.strip()
            lowered = clean.lower()
            if self._is_section_or_metadata_line(lowered):
                continue
            if 8 <= len(clean) <= 180 and not self._looks_like_author_line(clean):
                return clean
            if index + 1 < len(lines):
                combined = f"{clean} {lines[index + 1].strip()}"
                if 20 <= len(combined) <= 220 and not self._looks_like_author_line(combined):
                    return combined
        return ""

    def _is_section_or_metadata_line(self, lowered: str) -> bool:
        return bool(
            re.match(r"^(?:[a-z]{2}\s+)?(?:abstract|introduction|keywords|index terms)\b", lowered)
            or lowered.startswith(("proceedings", "published as"))
        )

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
        if text.count(",") < 2 or text.endswith((".", ":", ";")):
            return False
        scientific_markers = [
            "learning",
            "prediction",
            "representation",
            "survival",
            "dataset",
            "benchmark",
            "experiment",
            "evaluation",
            "tcga",
            "model",
            "method",
            "prognosis",
            "tissue",
            "graph",
            "prototype",
        ]
        if any(keyword in lowered for keyword in scientific_markers):
            return False
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return len(parts) >= 3 and all(len(part.split()) <= 4 for part in parts[:5])

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
        sentence_text = re.sub(r"\n+", ". ", text)
        for sentence in sentence_text.split("."):
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords) and len(sentence.strip()) > 40:
                return sentence.strip() + "."
        return ""

    def _best_keyword_sentence(self, text: str, keywords: list[str]) -> str:
        sentences = self._sentences(text)
        scored = []
        for index, sentence in enumerate(sentences):
            lowered = sentence.lower()
            hits = sum(1 for keyword in keywords if keyword in lowered)
            if not hits or len(sentence) < 45:
                continue
            score = hits * 10 - index
            if any(marker in lowered for marker in ["we propose", "in this paper", "we validate"]):
                score += 8
            if any(marker in lowered for marker in ["however", "neglect", "struggle"]):
                score += 5
            scored.append((score, sentence))
        if not scored:
            return ""
        return max(scored, key=lambda item: item[0])[1].rstrip(".") + "."

    def _sentences(self, text: str) -> list[str]:
        joined = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
        joined = re.sub(r"\s+", " ", joined).strip()
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", joined)
            if sentence.strip()
        ]

    def _guess_limitations(self, text: str, method_notes: str) -> list[str]:
        candidates = []
        direct = self._best_keyword_sentence(
            text,
            [
                "however",
                "neglect",
                "struggle",
                "limitation",
                "lack",
                "do not",
                "does not",
            ],
        )
        if direct:
            candidates.append(direct)
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
        lowered = text.lower()
        for token in [
            "representation",
            "attention",
            "optimization",
            "adaptation",
            "retrieval",
            "heterogeneous graph",
            "prototype learning",
            "weakly supervised learning",
            "survival prediction",
            "whole slide images",
        ]:
            if token in lowered and token not in terms:
                terms.append(token)
        if "prototype" in lowered and "prototype learning" not in terms:
            terms.append("prototype learning")
        if ("wsi" in lowered or "whole-slide" in lowered) and "whole slide images" not in terms:
            terms.append("whole slide images")
        return terms[:8]
