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

        artifacts = state.setdefault("artifacts", {})
        errors: dict[str, str] = {}
        error_details: list[dict[str, str]] = []
        candidates: list[CitationEntry] = []
        baseline_work: dict[str, Any] | None = None
        baseline = state.get("baseline")
        artifacts["related_work_baseline_mentioned_queries"] = []

        if baseline and baseline.title and baseline.title != "Baseline Paper":
            try:
                baseline_work = self._search_work(baseline.title)
            except Exception as exc:
                errors["baseline_search"] = str(exc)
                error_details.append(
                    self._error_detail(
                        source="baseline_search",
                        error=str(exc),
                        query=baseline.title,
                        sort="relevance_score:desc",
                    )
                )

        if baseline_work:
            try:
                candidates.extend(self._referenced_work_candidates(baseline_work, limit=3))
            except Exception as exc:
                errors["baseline_references"] = str(exc)
                references = [self._openalex_id(work_id) for work_id in baseline_work.get("referenced_works", [])]
                error_details.append(
                    self._error_detail(
                        source="baseline_references",
                        error=str(exc),
                        filter="openalex_id:" + "|".join([work_id for work_id in references if work_id][:25]),
                        sort="cited_by_count:desc",
                    )
                )
            try:
                candidates.extend(self._citing_work_candidates(baseline_work, limit=3))
            except Exception as exc:
                errors["baseline_citations"] = str(exc)
                error_details.append(
                    self._error_detail(
                        source="baseline_citations",
                        error=str(exc),
                        filter=f"cites:{self._openalex_id(str(baseline_work.get('id') or ''))}",
                        sort="publication_date:desc",
                    )
                )

        if baseline:
            mentioned_queries: list[str] = []
            try:
                mentioned_queries = self._mentioned_work_queries(baseline)
                artifacts["related_work_baseline_mentioned_queries"] = mentioned_queries
                candidates.extend(self._mentioned_work_candidates(baseline, limit=3, queries=mentioned_queries))
            except Exception as exc:
                errors["baseline_mentions"] = str(exc)
                error_details.append(
                    self._error_detail(
                        source="baseline_mentions",
                        error=str(exc),
                        query=self._compact_trace_value("; ".join(mentioned_queries[:3])),
                        sort="relevance_score:desc",
                    )
                )

        query = self._field_query(state)
        artifacts["related_work_field_query"] = query
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
                error_details.append(
                    self._error_detail(
                        source="influential_search",
                        error=str(exc),
                        query=query,
                        sort="cited_by_count:desc",
                    )
                )
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
                error_details.append(
                    self._error_detail(
                        source="recent_search",
                        error=str(exc),
                        query=query,
                        sort="publication_date:desc",
                    )
                )

        existing = self._resolve_baseline_seed_entries(state.get("bibliography", []), baseline_work)
        merged = self._merge_entries(existing, candidates)
        merged, pruned_seed_keys = self._prune_covered_seed_entries(merged, candidates)
        state["bibliography"] = merged
        artifacts["citation_keys"] = [entry.key for entry in merged]
        artifacts["related_work_discovery_mode"] = "openalex"
        artifacts["related_work_candidates"] = [
            self._candidate_artifact(entry) for entry in merged if "category=" in entry.note
        ]
        if pruned_seed_keys:
            artifacts["reference_pruned_seed_keys"] = pruned_seed_keys
        self._update_reference_verification(artifacts, merged)
        if errors:
            artifacts["related_work_discovery_errors"] = errors
        if error_details:
            artifacts["related_work_discovery_error_details"] = error_details
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

    def _mentioned_work_candidates(
        self,
        baseline,
        limit: int,
        *,
        queries: list[str] | None = None,
    ) -> list[CitationEntry]:
        queries = queries if queries is not None else self._mentioned_work_queries(baseline)
        entries: list[CitationEntry] = []
        for query in queries:
            if len(entries) >= limit:
                break
            surname, search_query = self._split_mentioned_query(query)
            data = self._query_openalex(
                {
                    "search": search_query,
                    "per-page": 5,
                    "sort": "relevance_score:desc",
                    "select": self._select_fields(),
                }
            )
            works = data.get("results", []) if isinstance(data, dict) else []
            selected = self._best_mentioned_work(surname, search_query, works)
            if not selected:
                continue
            entry = self._entry_from_work(
                selected,
                category="baseline_mentioned",
                note_prefix="Candidate discovered from a named work in the provided baseline related-work text",
            )
            entries.append(entry.model_copy(update={"query": query, "note": f"{entry.note} Source query: {query}."}))
        return entries

    def _mentioned_work_queries(self, baseline) -> list[str]:
        section_text = ""
        if getattr(baseline, "structured_sections", None):
            section_text = "\n".join(
                [
                    baseline.structured_sections.get("related_work", ""),
                    baseline.structured_sections.get("introduction", ""),
                ]
            )
        if not section_text:
            section_text = getattr(baseline, "extracted_text_preview", "")
        domain_context = self._baseline_query_context(baseline)
        references = getattr(baseline, "references", {}) or {}
        queries: list[str] = []
        for sentence in self._sentences(section_text):
            match = re.search(r"\b([A-Z][A-Za-z-]{2,})\s+et\s+al\.", sentence)
            if not match:
                continue
            surname = match.group(1)
            citation_match = re.search(r"\[(\d+)\]", sentence)
            reference = references.get(citation_match.group(1)) if citation_match else ""
            if reference:
                context = self._reference_query(surname, reference)
                if context:
                    queries.append(context)
                    if len(queries) >= 8:
                        break
                    continue
            context_tokens = (self._query_context(sentence) + " " + domain_context).split()
            context = " ".join(list(dict.fromkeys(context_tokens))[:12])
            if context:
                queries.append(f"{surname} {context}")
            if len(queries) >= 8:
                break
        return list(dict.fromkeys(queries))

    def _reference_query(self, surname: str, reference: str) -> str:
        compact = re.sub(r"\s+", " ", reference or "").strip()
        compact = re.sub(r"https?://\S+|doi:\S+|arXiv:\S+", " ", compact, flags=re.I)
        parts = [part.strip(" .") for part in re.split(r"\.\s+", compact) if part.strip(" .")]
        title = ""
        if len(parts) >= 2:
            title = parts[1]
        elif parts:
            title = parts[0]
        title = re.sub(r"\b(?:In|Proceedings|IEEE|ACM|Springer|PMLR)\b.*$", "", title).strip(" ,.;:")
        return f"{surname} | {title}"[:220].strip()

    def _split_mentioned_query(self, query: str) -> tuple[str, str]:
        if "|" in query:
            surname, search_query = query.split("|", 1)
            return surname.strip().split(maxsplit=1)[0], search_query.strip()
        return (query.split(maxsplit=1)[0] if query else ""), query

    def _baseline_query_context(self, baseline) -> str:
        parts = [getattr(baseline, "title", "")]
        parts.extend(getattr(baseline, "related_terms", []) or [])
        tokens: list[str] = []
        for part in parts:
            tokens.extend(self._title_terms(str(part)))
        unique_tokens = list(dict.fromkeys(tokens))
        priority = [
            token
            for token in unique_tokens
            if token
            in {
                "whole",
                "slide",
                "image",
                "survival",
                "predict",
                "cancer",
                "histology",
                "graph",
                "hypergraph",
                "pathology",
                "prognostic",
                "molecular",
                "outcome",
            }
        ]
        selected = priority + [token for token in unique_tokens if token not in priority]
        return " ".join(selected[:8])

    def _sentences(self, text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", text or "").strip()
        compact = re.sub(r"\bet\s+al\.", "et al<DOT>", compact)
        return [
            sentence.replace("et al<DOT>", "et al.").strip()
            for sentence in re.split(r"(?<=[.!?])\s+", compact)
            if len(sentence.strip()) >= 40
        ]

    def _query_context(self, sentence: str) -> str:
        cleaned = re.sub(r"\[[^\]]+\]", " ", sentence)
        cleaned = re.sub(r"\b[A-Z][A-Za-z-]{2,}\s+et\s+al\.", " ", cleaned)
        tokens = self._title_terms(cleaned)
        priority = [
            token
            for token in tokens
            if token
            in {
                "whole",
                "slide",
                "image",
                "survival",
                "predict",
                "cancer",
                "histology",
                "graph",
                "hypergraph",
                "weakly",
                "supervised",
                "pathology",
                "prognostic",
                "molecular",
                "outcome",
            }
        ]
        selected = priority[:8] if priority else tokens[:8]
        return " ".join(selected)

    def _best_mentioned_work(
        self,
        surname: str,
        query: str,
        works: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        author_matches = [work for work in works if self._work_has_author_surname(work, surname)]
        if not author_matches:
            return None
        relevant = [work for work in author_matches if self._mentioned_work_relevant(query, work)]
        return relevant[0] if relevant else None

    def _work_has_author_surname(self, work: dict[str, Any], surname: str) -> bool:
        expected = self._normalize_name_token(surname)
        if not expected:
            return False
        for author in self._authors(work):
            parts = [self._normalize_name_token(part) for part in re.findall(r"[A-Za-z-]+", author)]
            if expected in parts:
                return True
        return False

    def _mentioned_work_relevant(self, query: str, work: dict[str, Any]) -> bool:
        query_tokens = set(self._title_terms(query))
        title_tokens = set(self._title_terms(str(work.get("title") or "")))
        if not query_tokens or not title_tokens:
            return False
        return len(query_tokens & title_tokens) >= 2

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

    def _resolve_baseline_seed_entries(
        self,
        entries: list[CitationEntry],
        baseline_work: dict[str, Any] | None,
    ) -> list[CitationEntry]:
        if not baseline_work:
            return entries
        resolved = []
        for entry in entries:
            if self._baseline_seed_matches(entry, baseline_work):
                resolved.append(self._entry_from_baseline_work(entry, baseline_work))
            else:
                resolved.append(entry)
        return resolved

    def _baseline_seed_matches(self, entry: CitationEntry, baseline_work: dict[str, Any]) -> bool:
        if self._is_resolved(entry):
            return False
        work_title = str(baseline_work.get("title") or "")
        entry_text = " ".join([entry.key, entry.title, entry.query])
        return bool(
            work_title
            and entry_text.strip()
            and (
                self._title_key(work_title) == self._title_key(entry.title)
                or self._title_key(work_title) == self._title_key(entry.query)
                or entry.key.lower().startswith("baseline")
            )
        )

    def _entry_from_baseline_work(
        self,
        original: CitationEntry,
        work: dict[str, Any],
    ) -> CitationEntry:
        title = str(work.get("title") or original.title)
        doi = self._clean_doi(str(work.get("doi") or ""))
        return original.model_copy(
            update={
                "title": title,
                "authors": self._authors(work) or original.authors,
                "year": str(work.get("publication_year") or original.year),
                "venue": self._venue(work) or original.venue,
                "doi": doi,
                "url": self._work_url(work, doi),
                "note": (
                    "Resolved provided baseline paper during related-work discovery. "
                    "Verify relevance before submission."
                ),
            }
        )

    def _prune_covered_seed_entries(
        self,
        entries: list[CitationEntry],
        discovered: list[CitationEntry],
    ) -> tuple[list[CitationEntry], list[str]]:
        if not any(self._is_resolved(entry) for entry in discovered):
            return entries, []
        kept: list[CitationEntry] = []
        pruned: list[str] = []
        for entry in entries:
            if self._prunable_related_work_seed(entry):
                pruned.append(entry.key)
                continue
            kept.append(entry)
        return kept, pruned

    def _prunable_related_work_seed(self, entry: CitationEntry) -> bool:
        if not self._is_seed_entry(entry):
            return False
        note = entry.note.lower()
        authors = {author.lower() for author in entry.authors}
        return bool(
            "project keywords" in note
            or "seed related-work" in note
            or "related work authors" in authors
        )

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
            "query": entry.query,
            "category": category_match.group(1) if category_match else "unknown",
            "cited_by_count": int(cited_match.group(1)) if cited_match else 0,
        }

    def _update_reference_verification(
        self,
        artifacts: dict[str, Any],
        entries: list[CitationEntry],
    ) -> None:
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
        verification = {
            "resolved_count": len(resolved_keys),
            "unresolved_count": len(unresolved_seed_keys),
            "resolved_keys": resolved_keys,
            "unresolved_seed_keys": unresolved_seed_keys,
            "needs_manual_check_keys": needs_manual_check_keys,
        }
        artifacts["reference_verification"] = verification
        artifacts["reference_resolver_resolved"] = verification["resolved_count"]
        artifacts["reference_resolver_unresolved"] = verification["unresolved_count"]

    def _is_resolved(self, entry: CitationEntry) -> bool:
        has_metadata = bool(entry.year and entry.year != "TODO")
        has_real_authors = bool(
            entry.authors
            and not any(self._is_placeholder_author(author) for author in entry.authors)
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
        return author.lower() in {
            "baseline authors",
            "related work authors",
            "to be completed",
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

    def _error_detail(
        self,
        *,
        source: str,
        error: str,
        query: str = "",
        filter: str = "",
        sort: str = "",
    ) -> dict[str, str]:
        detail = {
            "source": source,
            "error": self._compact_trace_value(error),
        }
        if query:
            detail["query"] = self._compact_trace_value(query)
        if filter:
            detail["filter"] = self._compact_trace_value(filter)
        if sort:
            detail["sort"] = sort
        return detail

    def _compact_trace_value(self, value: str, limit: int = 240) -> str:
        compact = re.sub(r"\s+", " ", value or "").strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

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
            "method",
            "methods",
            "model",
            "models",
            "proposed",
            "employed",
            "used",
            "processing",
        }
        terms = []
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", title):
            normalized = self._normalize_topic_token(token)
            if normalized and normalized not in stopwords:
                terms.append(normalized)
        return terms

    def _normalize_topic_token(self, value: str) -> str:
        token = value.lower()
        aliases = {
            "images": "image",
            "outcomes": "outcome",
            "predicting": "predict",
            "prediction": "predict",
            "predictive": "predict",
            "predicted": "predict",
            "histological": "histology",
            "cnns": "cnn",
        }
        if token in aliases:
            return aliases[token]
        if token.endswith("ies") and len(token) > 5:
            return token[:-3] + "y"
        return token

    def _normalize_name_token(self, value: str) -> str:
        return re.sub(r"[^a-z]", "", value.lower())

    def _relevant_to_query(self, query: str, title: str) -> bool:
        query_tokens = set(self._title_terms(query))
        title_tokens = set(self._title_terms(title))
        if not query_tokens or not title_tokens:
            return False
        return len(query_tokens & title_tokens) >= 2
