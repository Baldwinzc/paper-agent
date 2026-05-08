"""Bibliography seed generation."""

from __future__ import annotations

import re
from collections import OrderedDict

from paper_agent.state import CitationEntry, PaperState


class BibliographyAgent:
    """Builds seed BibTeX entries and citation keys from local evidence."""

    BROAD_TERMS = {"representation", "attention", "optimization", "adaptation", "retrieval"}
    TECH_STOPWORDS = {
        "addresses",
        "adaptive",
        "analysis",
        "baseline",
        "confirmed",
        "contribution",
        "driven",
        "feature",
        "framework",
        "improvement",
        "innovation",
        "method",
        "minimal",
        "model",
        "notes",
        "online",
        "offline",
        "proposed",
        "regularization",
        "replacing",
        "setting",
        "static",
        "targeted",
        "training",
        "uses",
        "with",
    }

    def run(self, state: PaperState) -> PaperState:
        entries: OrderedDict[str, CitationEntry] = OrderedDict()
        baseline = state.get("baseline")
        request = state["request"]

        if baseline and baseline.title and baseline.title != "Baseline Paper":
            self._add_entry(
                entries,
                CitationEntry(
                    key=self._citation_key(baseline.title, preferred_prefix="baseline"),
                    title=baseline.title,
                    query=baseline.title,
                    authors=["Baseline authors"],
                    note="Seed entry extracted from the provided baseline PDF; verify metadata before submission.",
                ),
            )

        for term in self._research_threads(state):
            title = self._thread_title(term)
            self._add_entry(
                entries,
                CitationEntry(
                    key=self._citation_key(title, preferred_prefix=term),
                    title=title,
                    query=self._contextual_query(term, state),
                    authors=["Related work authors"],
                    note=(
                        "Seed related-work entry generated from project keywords; "
                        "replace with real paper metadata. Verify metadata before submission."
                    ),
                ),
            )

        if not entries:
            self._add_entry(
                entries,
                CitationEntry(
                    key="relatedworkseed",
                    title=f"Related work for {request.project_name}",
                    query=request.project_name,
                    authors=["To be completed"],
                    note="Placeholder bibliography seed; replace with real paper metadata. Verify metadata before submission.",
                ),
            )

        state["bibliography"] = list(entries.values())
        state.setdefault("artifacts", {})["citation_keys"] = [entry.key for entry in entries.values()]
        return state

    def _research_threads(self, state: PaperState) -> list[str]:
        request = state["request"]
        baseline = state.get("baseline")
        innovations = state.get("innovations", [])
        explicit_keywords = {self._normalize_term(term).lower() for term in request.keywords}
        terms: list[str] = []
        terms.extend(request.keywords)
        if baseline:
            terms.extend(baseline.related_terms)
        for innovation in innovations:
            terms.extend(self._innovation_research_terms(innovation))
        filtered = []
        for term in terms:
            normalized = self._normalize_term(term)
            if not normalized:
                continue
            if normalized.lower() in self.BROAD_TERMS and normalized.lower() not in explicit_keywords:
                continue
            filtered.append(normalized)
        return list(dict.fromkeys(filtered))[:8]

    def _innovation_research_terms(self, innovation) -> list[str]:
        text = self._strip_innovation_prefix(
            f"{innovation.name} {innovation.technical_idea}"
        )
        known_phrase = self._known_technical_phrase(text)
        if known_phrase:
            return [known_phrase]

        keyword_phrase = self._keyword_phrase(text)
        return [keyword_phrase] if keyword_phrase else []

    def _strip_innovation_prefix(self, text: str) -> str:
        return re.sub(r"\binnovation\s+\d+\s*:?", " ", text, flags=re.I)

    def _known_technical_phrase(self, text: str) -> str:
        lowered = text.lower()
        has_hypergraph = bool(re.search(r"\bhyper(?:edge|edges|graph|graphs)\b", lowered))
        has_ot = bool(re.search(r"\b(?:ot|optimal transport)\b", lowered))
        has_prototype = "prototype" in lowered
        has_reconstruction = "reconstruction" in lowered
        has_survival = "survival" in lowered

        if has_survival and has_reconstruction:
            return "survival prediction representation learning"
        if has_prototype and not has_hypergraph:
            return "prototype learning"

        terms: list[str] = []
        if has_ot:
            terms.append("optimal transport")
        if "wasserstein" in lowered and "barycenter" in lowered:
            terms.append("wasserstein barycenter")
        if has_hypergraph:
            terms.append("hypergraph learning")
        if has_prototype:
            terms.append("prototype learning")
        if has_survival:
            terms.append("survival prediction")
        if "cross-cluster" in lowered or "cross cluster" in lowered:
            terms.append("prototype clustering")

        deduped = list(dict.fromkeys(terms))
        return " ".join(deduped[:3])

    def _keyword_phrase(self, text: str) -> str:
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
            if token.lower() not in self.TECH_STOPWORDS
        ]
        deduped = list(dict.fromkeys(tokens))
        if len(deduped) < 2:
            return ""
        return " ".join(deduped[:4])

    def _contextual_query(self, term: str, state: PaperState) -> str:
        request = state["request"]
        normalized_term = self._normalize_term(term).lower()
        context_terms = [
            keyword
            for keyword in request.keywords
            if self._normalize_term(keyword).lower() != normalized_term
        ][:2]
        if not context_terms:
            context_terms = ["computer science"]
        return " ".join(dict.fromkeys(f"{term} {' '.join(context_terms)}".split()))

    def _thread_title(self, term: str) -> str:
        words = term.replace("-", " ").replace("_", " ").strip()
        return f"Representative work on {words}"

    def _add_entry(self, entries: OrderedDict[str, CitationEntry], entry: CitationEntry) -> None:
        key = entry.key
        suffix = 2
        while key in entries:
            key = f"{entry.key}{suffix}"
            suffix += 1
        entries[key] = entry.model_copy(update={"key": key})

    def _citation_key(self, title: str, preferred_prefix: str = "") -> str:
        base = self._normalize_term(preferred_prefix or title)
        tokens = re.findall(r"[a-zA-Z0-9]+", base)
        compact = "".join(tokens[:4]).lower()
        return compact[:32] or "citation"

    def _normalize_term(self, term: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9\s_-]+", " ", term)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
