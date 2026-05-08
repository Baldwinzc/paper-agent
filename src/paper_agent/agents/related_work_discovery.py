"""Scholarly related-work discovery."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from typing import Any

import httpx

from paper_agent.state import CitationEntry, PaperState


class RelatedWorkDiscoveryAgent:
    """Discovers classic, baseline-lineage, and recent related-work candidates."""

    OPENALEX_WORKS_URL = "https://api.openalex.org/works"

    def run(self, state: PaperState) -> PaperState:
        if self._disabled():
            state.setdefault("artifacts", {})["related_work_discovery_mode"] = "disabled"
            return state

        errors: dict[str, str] = {}
        candidates: list[CitationEntry] = []
        baseline_work: dict[str, Any] | None = None
        baseline = state.get("baseline")

        if baseline and baseline.title and baseline.title != "Baseline Paper":
            try:
                baseline_work = self._search_work(baseline.title)
            except Exception as exc:
                errors["baseline_search"] = str(exc)

        if baseline_work:
            try:
                candidates.extend(self._referenced_work_candidates(baseline_work, limit=3))
            except Exception as exc:
                errors["baseline_references"] = str(exc)
            try:
                candidates.extend(self._citing_work_candidates(baseline_work, limit=3))
            except Exception as exc:
                errors["baseline_citations"] = str(exc)

        query = self._field_query(state)
        if query:
            try:
                candidates.extend(
                    self._search_candidates(
                        query,
                        category="influential",
                        note_prefix="High-citation candidate discovered from field query",
                        sort="cited_by_count:desc",
                        limit=3,
                    )
                )
            except Exception as exc:
                errors["influential_search"] = str(exc)
            try:
                candidates.extend(
                    self._search_candidates(
                        query,
                        category="recent",
                        note_prefix="Recent candidate discovered from field query",
                        sort="publication_date:desc",
                        limit=3,
                    )
                )
            except Exception as exc:
                errors["recent_search"] = str(exc)

        merged = self._merge_entries(state.get("bibliography", []), candidates)
        state["bibliography"] = merged
        artifacts = state.setdefault("artifacts", {})
        artifacts["citation_keys"] = [entry.key for entry in merged]
        artifacts["related_work_discovery_mode"] = "openalex"
        artifacts["related_work_candidates"] = [
            self._candidate_artifact(entry) for entry in merged if "category=" in entry.note
        ]
        if errors:
            artifacts["related_work_discovery_errors"] = errors
        return state

    def _disabled(self) -> bool:
        explicit = os.getenv("PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY", "").strip().lower()
        if explicit in {"1", "true", "yes", "on"}:
            return True
        reference_disabled = os.getenv("PAPER_AGENT_DISABLE_REFERENCE_RESOLVE", "").strip().lower()
        return reference_disabled in {"1", "true", "yes", "on"}

    def _search_work(self, query: str) -> dict[str, Any] | None:
        data = self._query_openalex(
            {
                "search": query,
                "per-page": 1,
                "sort": "relevance_score:desc",
                "select": self._select_fields(),
            }
        )
        results = data.get("results", []) if isinstance(data, dict) else []
        return results[0] if results else None

    def _referenced_work_candidates(self, baseline_work: dict[str, Any], limit: int) -> list[CitationEntry]:
        references = [self._openalex_id(work_id) for work_id in baseline_work.get("referenced_works", [])]
        references = [work_id for work_id in references if work_id][:25]
        if not references:
            return []
        data = self._query_openalex(
            {
                "filter": "openalex_id:" + "|".join(references),
                "per-page": min(25, len(references)),
                "sort": "cited_by_count:desc",
                "select": self._select_fields(),
            }
        )
        works = data.get("results", []) if isinstance(data, dict) else []
        return [
            self._entry_from_work(
                work,
                category="baseline_reference",
                note_prefix="Baseline-reference candidate cited by the provided baseline paper",
            )
            for work in works[:limit]
        ]

    def _citing_work_candidates(self, baseline_work: dict[str, Any], limit: int) -> list[CitationEntry]:
        baseline_id = self._openalex_id(str(baseline_work.get("id") or ""))
        if not baseline_id:
            return []
        data = self._query_openalex(
            {
                "filter": f"cites:{baseline_id}",
                "per-page": limit,
                "sort": "publication_date:desc",
                "select": self._select_fields(),
            }
        )
        works = data.get("results", []) if isinstance(data, dict) else []
        return [
            self._entry_from_work(
                work,
                category="baseline_citing",
                note_prefix="Recent follow-up candidate that cites the provided baseline paper",
            )
            for work in works[:limit]
        ]

    def _search_candidates(
        self,
        query: str,
        category: str,
        note_prefix: str,
        sort: str,
        limit: int,
    ) -> list[CitationEntry]:
        data = self._query_openalex(
            {
                "filter": f"title_and_abstract.search:{query}",
                "per-page": max(limit * 5, 10),
                "sort": sort,
                "select": self._select_fields(),
            }
        )
        works = data.get("results", []) if isinstance(data, dict) else []
        relevant = [work for work in works if self._relevant_to_query(query, str(work.get("title") or ""))]
        selected = relevant[:limit] if relevant else works[:limit]
        return [
            self._entry_from_work(work, category=category, note_prefix=note_prefix)
            for work in selected
        ]

    def _query_openalex(self, params: dict[str, str | int]) -> dict[str, Any]:
        request_params = dict(params)
        mailto = os.getenv("OPENALEX_MAILTO", "").strip()
        if mailto:
            request_params["mailto"] = mailto
        response = httpx.get(self.OPENALEX_WORKS_URL, params=request_params, timeout=20)
        response.raise_for_status()
        return response.json()

    def _entry_from_work(
        self,
        work: dict[str, Any],
        category: str,
        note_prefix: str,
    ) -> CitationEntry:
        title = str(work.get("title") or "Untitled related work")
        doi = self._clean_doi(str(work.get("doi") or ""))
        cited_by_count = int(work.get("cited_by_count") or 0)
        return CitationEntry(
            key=self._citation_key(title),
            title=title,
            query=title,
            authors=self._authors(work),
            year=str(work.get("publication_year") or ""),
            venue=self._venue(work),
            doi=doi,
            url=self._work_url(work, doi),
            note=(
                f"{note_prefix}; category={category}; cited_by_count={cited_by_count}. "
                "Verify relevance before submission."
            ),
        )

    def _merge_entries(
        self,
        existing: list[CitationEntry],
        discovered: list[CitationEntry],
    ) -> list[CitationEntry]:
        entries: OrderedDict[str, CitationEntry] = OrderedDict()
        seen_dois = {entry.doi.lower().strip() for entry in existing if entry.doi}
        seen_titles = {self._title_key(entry.title) for entry in existing if entry.title}
        for entry in existing:
            entries[entry.key] = entry
        for entry in discovered:
            title_key = self._title_key(entry.title)
            doi_key = entry.doi.lower().strip()
            if (doi_key and doi_key in seen_dois) or title_key in seen_titles:
                continue
            key = entry.key
            suffix = 2
            while key in entries:
                key = f"{entry.key}{suffix}"
                suffix += 1
            entries[key] = entry.model_copy(update={"key": key})
            if doi_key:
                seen_dois.add(doi_key)
            seen_titles.add(title_key)
        return list(entries.values())

    def _candidate_artifact(self, entry: CitationEntry) -> dict[str, Any]:
        category_match = re.search(r"category=([^;]+)", entry.note)
        cited_match = re.search(r"cited_by_count=(\d+)", entry.note)
        return {
            "key": entry.key,
            "title": entry.title,
            "authors": entry.authors,
            "year": entry.year,
            "venue": entry.venue,
            "doi": entry.doi,
            "url": entry.url,
            "category": category_match.group(1) if category_match else "unknown",
            "cited_by_count": int(cited_match.group(1)) if cited_match else 0,
        }

    def _field_query(self, state: PaperState) -> str:
        request = state["request"]
        baseline = state.get("baseline")
        terms: list[str] = []
        if request.keywords:
            terms.extend(request.keywords[:4])
        elif baseline:
            terms.extend(baseline.related_terms[:4])
            terms.extend(self._title_terms(baseline.title)[:4])
        if not terms:
            terms.append(request.project_name)
        query = " ".join(dict.fromkeys(term for term in terms if term))
        return query.replace("-", " ")[:220]

    def _select_fields(self) -> str:
        return ",".join(
            [
                "id",
                "doi",
                "title",
                "publication_year",
                "publication_date",
                "authorships",
                "primary_location",
                "ids",
                "referenced_works",
                "cited_by_count",
            ]
        )

    def _authors(self, work: dict[str, Any]) -> list[str]:
        authors = []
        for authorship in work.get("authorships", [])[:8]:
            author = authorship.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(str(name))
        return authors

    def _venue(self, work: dict[str, Any]) -> str:
        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        return str(source.get("display_name") or "")

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
        return str(ids.get("openalex") or work.get("id") or "")

    def _openalex_id(self, value: str) -> str:
        return value.rstrip("/").rsplit("/", 1)[-1] if value else ""

    def _citation_key(self, title: str) -> str:
        tokens = re.findall(r"[a-zA-Z0-9]+", title.lower())
        return "".join(tokens[:4])[:32] or "relatedwork"

    def _title_key(self, title: str) -> str:
        return " ".join(re.findall(r"[a-zA-Z0-9]+", title.lower()))

    def _title_terms(self, title: str) -> list[str]:
        title = title.replace("-", " ")
        stopwords = {
            "and",
            "for",
            "from",
            "into",
            "the",
            "with",
            "using",
            "based",
            "paper",
        }
        return [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", title)
            if token.lower() not in stopwords
        ]

    def _relevant_to_query(self, query: str, title: str) -> bool:
        query_tokens = set(self._title_terms(query))
        title_tokens = set(self._title_terms(title))
        if not query_tokens or not title_tokens:
            return False
        return len(query_tokens & title_tokens) >= 2
