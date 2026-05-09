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
    S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self) -> None:
        self._semantic_scholar_rate_limited = False

    def run(self, state: PaperState) -> PaperState:
        if self._disabled():
            state.setdefault("artifacts", {})["reference_resolver_mode"] = "disabled"
            return state

        artifacts = state.setdefault("artifacts", {})
        resolved: list[CitationEntry] = []
        errors: dict[str, str] = {}
        trace: list[dict[str, object]] = []
        for entry in state.get("bibliography", []):
            try:
                resolved_entry = self._resolve_entry(entry)
            except Exception as exc:
                errors[entry.key] = str(exc)
                resolved_entry = entry
            resolved.append(resolved_entry)
            trace.append(self._trace_item(entry, resolved_entry, errors.get(entry.key, "")))

        resolved = self._deduplicate_by_doi(resolved, state)
        state["bibliography"] = resolved
        verification = self._verification_summary(resolved)
        artifacts["reference_resolver_mode"] = "openalex"
        artifacts["reference_resolver_resolved"] = verification["resolved_count"]
        artifacts["reference_resolver_unresolved"] = verification["unresolved_count"]
        artifacts["reference_verification"] = verification
        artifacts["citation_keys"] = [entry.key for entry in resolved]
        artifacts["reference_resolution_trace"] = self._annotate_trace_with_deduplication(
            trace,
            artifacts.get("citation_key_aliases", {}),
        )
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
            return self._resolve_entry_with_semantic_scholar(
                entry,
                query,
                f"{entry.note} OpenAlex returned no candidates for query: {query}.",
            )

        work = self._best_openalex_work(query, results)
        if not work:
            return self._resolve_entry_with_semantic_scholar(
                entry,
                query,
                (
                    f"{entry.note} OpenAlex candidate was rejected as low-confidence "
                    f"for query: {query}."
                ),
            )
        return self._entry_from_openalex(entry, query, work)

    def _query_openalex(self, query: str) -> dict[str, Any]:
        params: dict[str, str | int] = {
            "search": query,
            "per-page": 5,
            "select": "doi,title,publication_year,authorships,primary_location,ids",
        }
        mailto = os.getenv("OPENALEX_MAILTO", "").strip()
        if mailto:
            params["mailto"] = mailto
        response = httpx.get(self.OPENALEX_WORKS_URL, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def _best_openalex_work(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        candidates = [
            (self._match_score(query, str(work.get("title") or "")), index, work)
            for index, work in enumerate(results)
            if isinstance(work, dict)
        ]
        confident = [candidate for candidate in candidates if candidate[0] >= 2.0]
        if not confident:
            return None
        confident.sort(key=lambda item: (-item[0], item[1]))
        return confident[0][2]

    def _resolve_entry_with_semantic_scholar(
        self,
        entry: CitationEntry,
        query: str,
        prior_note: str,
    ) -> CitationEntry:
        if self._semantic_scholar_rate_limited:
            return entry.model_copy(
                update={"note": f"{prior_note} Semantic Scholar fallback skipped after rate limit."}
            )
        try:
            data = self._query_semantic_scholar(query)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self._semantic_scholar_rate_limited = True
                return entry.model_copy(
                    update={
                        "note": (
                            f"{prior_note} Semantic Scholar fallback was rate limited "
                            "and skipped for the remaining references."
                        )
                    }
                )
            return entry.model_copy(
                update={
                    "note": (
                        f"{prior_note} Semantic Scholar lookup failed with HTTP "
                        f"{exc.response.status_code}."
                    )
                }
            )
        except Exception as exc:
            return entry.model_copy(
                update={"note": f"{prior_note} Semantic Scholar lookup failed: {type(exc).__name__}."}
            )
        results = data.get("data", []) if isinstance(data, dict) else []
        if not results:
            return entry.model_copy(
                update={"note": f"{prior_note} Semantic Scholar returned no candidates."}
            )
        paper = results[0]
        if not self._confident_match(query, str(paper.get("title") or "")):
            return entry.model_copy(
                update={"note": f"{prior_note} Semantic Scholar candidate was low-confidence."}
            )
        return self._entry_from_semantic_scholar(entry, query, paper)

    def _query_semantic_scholar(self, query: str) -> dict[str, Any]:
        params = {
            "query": query,
            "limit": 1,
            "fields": "title,year,venue,authors,externalIds,url",
        }
        headers = {}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if api_key:
            headers["x-api-key"] = api_key
        response = httpx.get(self.S2_SEARCH_URL, params=params, headers=headers, timeout=15)
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

    def _entry_from_semantic_scholar(
        self,
        original: CitationEntry,
        query: str,
        paper: dict[str, Any],
    ) -> CitationEntry:
        external_ids = paper.get("externalIds") or {}
        doi = self._clean_doi(str(external_ids.get("DOI") or ""))
        title = str(paper.get("title") or original.title)
        authors = [
            str(author.get("name"))
            for author in paper.get("authors", [])[:8]
            if author.get("name")
        ]
        note = f"Resolved by Semantic Scholar from query: {query}. Verify relevance before submission."
        return original.model_copy(
            update={
                "title": title,
                "authors": authors or original.authors,
                "year": str(paper.get("year") or original.year),
                "venue": str(paper.get("venue") or original.venue),
                "doi": doi,
                "url": f"https://doi.org/{doi}" if doi else str(paper.get("url") or original.url),
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

    def _trace_item(
        self,
        original: CitationEntry,
        resolved: CitationEntry,
        error: str = "",
    ) -> dict[str, object]:
        return {
            "key": original.key,
            "query": original.query or original.title,
            "input_title": original.title,
            "resolved_title": resolved.title,
            "status": self._resolution_status(resolved, error),
            "source": self._resolution_source(resolved),
            "year": resolved.year,
            "venue": resolved.venue,
            "doi": resolved.doi,
            "url": resolved.url,
            "authors": resolved.authors[:5],
            "note": resolved.note,
            "error": error,
        }

    def _resolution_status(self, entry: CitationEntry, error: str = "") -> str:
        if error:
            return "error"
        if self._is_resolved(entry):
            return "resolved"
        if self._is_seed_entry(entry):
            return "unresolved_seed"
        return "needs_manual_check"

    def _resolution_source(self, entry: CitationEntry) -> str:
        note = entry.note.lower()
        if "openalex" in note:
            return "openalex"
        if "semantic scholar" in note:
            return "semantic_scholar"
        return "seed"

    def _annotate_trace_with_deduplication(
        self,
        trace: list[dict[str, object]],
        aliases: dict[str, str],
    ) -> list[dict[str, object]]:
        annotated = []
        for item in trace:
            key = str(item.get("key", ""))
            retained_key = aliases.get(key, key)
            annotated.append(
                {
                    **item,
                    "retained": retained_key == key,
                    "retained_key": retained_key,
                }
            )
        return annotated

    def _verification_summary(self, entries: list[CitationEntry]) -> dict[str, object]:
        resolved_keys = []
        unresolved_seed_keys = []
        needs_manual_check_keys = []
        for entry in entries:
            if self._is_resolved(entry):
                resolved_keys.append(entry.key)
                needs_manual_check_keys.append(entry.key)
            elif self._is_seed_entry(entry):
                unresolved_seed_keys.append(entry.key)
            else:
                needs_manual_check_keys.append(entry.key)
        return {
            "resolved_count": len(resolved_keys),
            "unresolved_count": len(unresolved_seed_keys),
            "resolved_keys": resolved_keys,
            "unresolved_seed_keys": unresolved_seed_keys,
            "needs_manual_check_keys": needs_manual_check_keys,
        }

    def _is_resolved(self, entry: CitationEntry) -> bool:
        has_metadata = bool(entry.year and entry.year != "TODO")
        has_real_authors = bool(
            entry.authors
            and not any(
                self._is_placeholder_author(author)
                for author in entry.authors
            )
        )
        return bool(entry.doi or (has_metadata and has_real_authors))

    def _is_seed_entry(self, entry: CitationEntry) -> bool:
        note = entry.note.lower()
        return (
            "seed" in note
            or "placeholder" in note
            or "replace with real" in note
            or any(self._is_placeholder_author(author) for author in entry.authors)
        )

    def _is_placeholder_author(self, author: str) -> bool:
        lowered = author.lower()
        return lowered in {
            "baseline authors",
            "related work authors",
            "to be completed",
        }

    def _confident_match(self, query: str, title: str) -> bool:
        return self._match_score(query, title) >= 2.0

    def _match_score(self, query: str, title: str) -> float:
        query_tokens = self._tokens(query)
        title_tokens = self._tokens(title)
        if not query_tokens or not title_tokens:
            return 0.0
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
        score = float(len(overlap))
        if specific_tokens & title_tokens:
            score += 0.5
        if len(overlap) >= max(2, len(specific_tokens) // 2):
            score += 0.5
        if len(overlap) >= 3:
            return score
        if bool(specific_tokens & title_tokens) and len(overlap) >= 2:
            return score
        return 0.0

    def _tokens(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(token) > 2
        }
