"""Reviewer simulation."""

from __future__ import annotations

import re

from paper_agent.state import PaperState, ReviewFinding


class ReviewerAgent:
    """Flags risks before the draft is treated as paper-ready."""

    def run(self, state: PaperState) -> PaperState:
        findings: list[ReviewFinding] = []
        experiments = state.get("experiments")
        innovations = state.get("innovations", [])
        sections = state.get("sections")

        for innovation in innovations:
            if "Needs manual confirmation" in innovation.risk or "Insufficient" in innovation.risk:
                findings.append(
                    ReviewFinding(
                        severity="major",
                        issue=f"{innovation.name} is not fully validated.",
                        suggestion="Confirm novelty and evidence before finalizing the Method section.",
                    )
                )

        if experiments and experiments.missing_details:
            findings.append(
                ReviewFinding(
                    severity="major",
                    issue="Experiment section lacks required details.",
                    suggestion="Fill in datasets, metrics, baselines, and exact result tables.",
                )
            )

        if sections and innovations:
            traceability = self._innovation_traceability(sections.method, innovations)
            state.setdefault("artifacts", {})["innovation_traceability"] = traceability
            omitted = [item for item in traceability if not item["mentioned_in_method"]]
            if omitted:
                findings.append(
                    ReviewFinding(
                        severity="major",
                        issue=(
                            "Method section omits innovation points: "
                            + ", ".join(item["name"] for item in omitted[:3])
                            + "."
                        ),
                        suggestion="Revise Method so each accepted innovation point has a concrete subsection or paragraph.",
                    )
                )

        if sections and "Citation placeholders" in sections.related_work:
            findings.append(
                ReviewFinding(
                    severity="minor",
                    issue="Related Work still uses citation placeholders.",
                    suggestion="Add bibliography ingestion and replace placeholders with BibTeX keys.",
                )
            )

        if state.get("bibliography"):
            unresolved = [
                entry
                for entry in state["bibliography"]
                if self._unresolved_seed_entry(entry)
            ]
            if unresolved:
                keys = ", ".join(entry.key for entry in unresolved[:5])
                findings.append(
                    ReviewFinding(
                        severity="minor",
                        issue=(
                            f"Bibliography contains {len(unresolved)} unresolved seed "
                            f"entries: {keys}."
                        ),
                        suggestion=(
                            "Resolve these generated seeds to real paper metadata or remove them "
                            "before treating the bibliography as submission-ready."
                        ),
                    )
                )

        if sections:
            coverage = self._related_work_citation_coverage(
                sections.related_work,
                state.get("bibliography", []),
                state.get("artifacts", {}).get("citation_key_aliases", {}),
            )
            state.setdefault("artifacts", {})["related_work_citation_coverage"] = coverage
            missing_coverage = [
                item
                for item in coverage
                if item["requires_citation"] and not item["covered_by_real_citation"]
            ]
            if missing_coverage:
                threads = ", ".join(item["thread"] for item in missing_coverage[:5])
                findings.append(
                    ReviewFinding(
                        severity="minor",
                        issue=f"Related Work threads lack real citation coverage: {threads}.",
                        suggestion=(
                            "Add verified citations to each research-thread subsection or remove "
                            "unsupported related-work claims."
                        ),
                    )
                )

            placeholder_hits = self._placeholder_hits(sections.model_dump())
            if placeholder_hits:
                findings.append(
                    ReviewFinding(
                        severity="minor",
                        issue=f"Draft still contains placeholders: {', '.join(placeholder_hits[:5])}.",
                        suggestion="Replace figure/table/detail placeholders before treating the draft as paper-ready.",
                    )
                )

            outline_hits = self._outline_language_hits(sections.model_dump())
            state.setdefault("artifacts", {})["outline_language_hits"] = outline_hits
            if outline_hits:
                findings.append(
                    ReviewFinding(
                        severity="minor",
                        issue=(
                            "Draft still contains outline or procedural language in "
                            + ", ".join(outline_hits[:5])
                            + "."
                        ),
                        suggestion="Rewrite these passages as paper prose rather than instructions to the writer.",
                    )
                )

        state["review_findings"] = [*state.get("review_findings", []), *findings]
        return state

    def _placeholder_hits(self, section_values: dict[str, str]) -> list[str]:
        patterns = [
            r"\bTODO\b",
            r"\bTBD\b",
            r"\bplaceholder\b",
            r"Table\s+[XY]",
            r"Fig\.\s*\[",
            r"\[.*?to be (?:added|filled|completed|refined).*?\]",
        ]
        hits = []
        for section, text in section_values.items():
            for pattern in patterns:
                if re.search(pattern, text, flags=re.I):
                    hits.append(section)
                    break
        return hits

    def _unresolved_seed_entry(self, entry) -> bool:
        if entry.doi or (entry.year and entry.year != "TODO" and self._has_real_author(entry)):
            return False
        note = entry.note.lower()
        return (
            "seed" in note
            or "placeholder" in note
            or "replace with real" in note
            or not entry.year
            or not self._has_real_author(entry)
        )

    def _has_real_author(self, entry) -> bool:
        placeholder_authors = {
            "baseline authors",
            "related work authors",
            "to be completed",
        }
        return bool(
            entry.authors
            and not any(author.lower() in placeholder_authors for author in entry.authors)
        )

    def _related_work_citation_coverage(
        self,
        related_work: str,
        bibliography,
        citation_aliases: dict[str, str],
    ) -> list[dict[str, object]]:
        if not related_work.strip():
            return []

        real_keys = {
            entry.key
            for entry in bibliography
            if not self._unresolved_seed_entry(entry)
        }
        coverage = []
        for thread, body in self._related_work_threads(related_work):
            citation_keys = self._citation_keys(body, citation_aliases)
            real_citation_keys = [key for key in citation_keys if key in real_keys]
            requires_citation = self._thread_requires_citation(thread)
            coverage.append(
                {
                    "thread": thread,
                    "requires_citation": requires_citation,
                    "citation_keys": citation_keys,
                    "real_citation_keys": real_citation_keys,
                    "covered_by_real_citation": bool(real_citation_keys) or not requires_citation,
                }
            )
        return coverage

    def _related_work_threads(self, related_work: str) -> list[tuple[str, str]]:
        threads: list[tuple[str, str]] = []
        current_heading = "Related Work"
        current_lines: list[str] = []

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                threads.append((current_heading, body))

        for line in related_work.splitlines():
            if line.startswith("### "):
                flush()
                current_heading = line[4:].strip() or "Related Work"
                current_lines = []
            else:
                current_lines.append(line)
        flush()
        return threads

    def _citation_keys(self, text: str, citation_aliases: dict[str, str]) -> list[str]:
        raw_keys: list[str] = []
        raw_keys.extend(
            key.strip()
            for match in re.finditer(r"\\cite\{([A-Za-z0-9_,\s-]+)\}", text)
            for key in match.group(1).split(",")
        )
        raw_keys.extend(
            key.strip()
            for match in re.finditer(r"\[([A-Za-z0-9_,\s-]+)\]", text)
            for key in match.group(1).split(",")
        )
        normalized = [
            citation_aliases.get(key, key)
            for key in raw_keys
            if key
        ]
        return list(dict.fromkeys(normalized))

    def _thread_requires_citation(self, thread: str) -> bool:
        lowered = thread.lower()
        optional_markers = {
            "relation to the proposed method",
            "relation to proposed method",
            "proposed method",
            "our method",
            "contribution positioning",
        }
        return not any(marker in lowered for marker in optional_markers)

    def _outline_language_hits(self, section_values: dict[str, str]) -> list[str]:
        patterns = [
            r"\b(?:introduction|related work|experiments?|conclusion|method)\s+section\s+should\b",
            r"\bshould\s+(?:open|include|discuss|state|restate|summarize|explain)\b",
            r"\bshould\s+be\s+(?:made|treated|discussed|positioned|resolved|filled|refined)\b",
            r"\bto be refined\b",
            r"\bcurrent missing details\b",
            r"\bcurrent parsed result summary\b",
            r"\bfinal conclusion should\b",
        ]
        hits = []
        for section, text in section_values.items():
            for pattern in patterns:
                if re.search(pattern, text, flags=re.I):
                    hits.append(section)
                    break
        return hits

    def _innovation_traceability(self, method_text: str, innovations) -> list[dict[str, object]]:
        return [
            {
                "name": innovation.name,
                "mentioned_in_method": self._innovation_mentioned(method_text, innovation),
                "evidence_count": len(innovation.evidence),
                "evidence_preview": innovation.evidence[:2],
            }
            for innovation in innovations
        ]

    def _innovation_mentioned(self, method_text: str, innovation) -> bool:
        lowered_method = method_text.lower()
        normalized_name = re.sub(r"^innovation\s+\d+:\s*", "", innovation.name, flags=re.I).lower()
        if normalized_name and normalized_name in lowered_method:
            return True

        tokens = self._content_tokens(f"{innovation.name} {innovation.technical_idea}")
        if not tokens:
            return False
        hits = sum(1 for token in tokens if token in lowered_method)
        return hits >= min(3, len(tokens))

    def _content_tokens(self, text: str) -> list[str]:
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "into",
            "that",
            "this",
            "method",
            "innovation",
        }
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
            if token.lower() not in stopwords
        ]
        return list(dict.fromkeys(tokens))[:8]
