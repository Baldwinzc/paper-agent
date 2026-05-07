"""Resolve bibliography seed entries with public scholarly metadata."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from paper_agent.state import CitationEntry, PaperState


class ReferenceResolverAgent:
    """Uses OpenAlex to enrich seed bibliography entries."""

    OPENALEX_WORKS_URL = "https://api.openalex.org/works"

    def run(self, state: PaperState) -> PaperState:
        if self._disabled():
            state.setdefault("artifacts", {})["reference_resolver_mode"] = "disabled"
            return state

        resolved: list[CitationEntry] = []
        errors: dict[str, str] = {}
        for entry in state.get("bibliography", []):
            try:
                resolved.append(self._resolve_entry(entry))
            except Exception as exc:
                errors[entry.key] = str(exc)
                resolved.append(entry)

        resolved = self._deduplicate_by_doi(resolved, state)
        state["bibliography"] = resolved
        artifacts = state.setdefault("artifacts", {})
        artifacts["reference_resolver_mode"] = "openalex"
        artifacts["reference_resolver_resolved"] = sum(1 for entry in resolved if entry.year or entry.doi)
        artifacts["citation_keys"] = [entry.key for entry in resolved]
        if errors:
            artifacts["reference_resolver_errors"] = errors
        return state

    def _disabled(self) -> bool:
        value = os.getenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _resolve_entry(self, entry: CitationEntry) -> CitationEntry:
        query = entry.query or entry.title
        if not query.strip():
            return entry

        data = self._query_openalex(query)
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            return entry.model_copy(
                update={"note": f"{entry.note} OpenAlex returned no candidates for query: {query}."}
            )

        work = results[0]
        if not self._confident_match(query, str(work.get("title") or "")):
            return entry.model_copy(
                update={
                    "note": (
                        f"{entry.note} OpenAlex candidate was rejected as low-confidence "
                        f"for query: {query}."
                    )
                }
            )
        return self._entry_from_openalex(entry, query, work)

    def _query_openalex(self, query: str) -> dict[str, Any]:
        params: dict[str, str | int] = {
            "search": query,
            "per-page": 1,
            "select": "doi,title,publication_year,authorships,primary_location,ids",
        }
        mailto = os.getenv("OPENALEX_MAILTO", "").strip()
        if mailto:
            params["mailto"] = mailto
        response = httpx.get(self.OPENALEX_WORKS_URL, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def _entry_from_openalex(
        self,
        original: CitationEntry,
        query: str,
        work: dict[str, Any],
    ) -> CitationEntry:
        title = str(work.get("title") or original.title)
        doi = self._clean_doi(str(work.get("doi") or ""))
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        venue = str(source.get("display_name") or original.venue)
        url = self._work_url(work, doi)
        authors = self._authors(work)
        year = str(work.get("publication_year") or original.year)
        note = f"Resolved by OpenAlex from query: {query}. Verify relevance before submission."

        return original.model_copy(
            update={
                "title": title,
                "authors": authors or original.authors,
                "year": year,
                "venue": venue,
                "doi": doi,
                "url": url,
                "note": note,
            }
        )

    def _authors(self, work: dict[str, Any]) -> list[str]:
        authors = []
        for authorship in work.get("authorships", [])[:8]:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(str(name))
        return authors

    def _clean_doi(self, doi: str) -> str:
        doi = doi.strip()
        for prefix in ["https://doi.org/", "http://doi.org/"]:
            if doi.lower().startswith(prefix):
                return doi[len(prefix) :]
        return doi

    def _work_url(self, work: dict[str, Any], doi: str) -> str:
        if doi:
            return f"https://doi.org/{doi}"
        ids = work.get("ids") or {}
        return str(ids.get("openalex") or "")

    def _deduplicate_by_doi(
        self,
        entries: list[CitationEntry],
        state: PaperState,
    ) -> list[CitationEntry]:
        seen: dict[str, str] = {}
        deduped: list[CitationEntry] = []
        aliases: dict[str, str] = {}
        for entry in entries:
            doi = entry.doi.lower().strip()
            if doi and doi in seen:
                aliases[entry.key] = seen[doi]
                continue
            if doi:
                seen[doi] = entry.key
            deduped.append(entry)
        if aliases:
            state.setdefault("artifacts", {})["citation_key_aliases"] = aliases
        return deduped

    def _confident_match(self, query: str, title: str) -> bool:
        query_tokens = self._tokens(query)
        title_tokens = self._tokens(title)
        if not query_tokens or not title_tokens:
            return False
        overlap = query_tokens & title_tokens
        specific_tokens = {
            token
            for token in query_tokens
            if token
            not in {
                "a",
                "an",
                "and",
                "for",
                "in",
                "of",
                "on",
                "the",
                "with",
                "learning",
                "prediction",
                "study",
            }
        }
        if len(overlap) >= 3:
            return True
        return bool(specific_tokens & title_tokens) and len(overlap) >= 2

    def _tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(token) > 2
        }
