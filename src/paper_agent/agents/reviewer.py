"""Reviewer simulation."""

from __future__ import annotations

import re

from paper_agent.state import PaperState, ReviewFinding


class ReviewerAgent:
    """Flags risks before the draft is treated as paper-ready."""

    METRIC_ALIASES = {
        "ACC": "ACCURACY",
        "ACCURACY": "ACCURACY",
        "AUC": "AUC",
        "BLEU": "BLEU",
        "BRIER": "BRIER SCORE",
        "BRIER SCORE": "BRIER SCORE",
        "C INDEX": "C-INDEX",
        "C-INDEX": "C-INDEX",
        "CONCORDANCE": "C-INDEX",
        "CONCORDANCE INDEX": "C-INDEX",
        "F1": "F1",
        "IBS": "IBS",
        "MAP": "MAP",
        "MAE": "MAE",
        "MIOU": "MIOU",
        "MRR": "MRR",
        "PSNR": "PSNR",
        "RMSE": "RMSE",
        "ROUGE": "ROUGE",
    }
    METRIC_PATTERN = (
        r"\b(c-index|concordance(?: index)?|ibs|brier(?: score)?|acc(?:uracy)?|"
        r"f1|auc|mrr|map|bleu|rouge|mae|rmse|psnr|miou)\b"
    )
    DATASET_STOPWORDS = {
        "API",
        "AUC",
        "BHE",
        "CLI",
        "CPU",
        "GPU",
        "IBS",
        "IEEE",
        "INDEX",
        "LLM",
        "OT",
        "PDF",
        "TPAMI",
        "WSI",
        "WSIS",
    }

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

        if sections:
            consistency = self._factual_consistency_checks(sections, experiments, innovations)
            state.setdefault("artifacts", {})["factual_consistency"] = consistency
            consistency_issues = [
                item
                for item in consistency
                if item["status"] == "needs_review"
            ]
            if consistency_issues:
                issue_names = ", ".join(item["check"] for item in consistency_issues[:5])
                findings.append(
                    ReviewFinding(
                        severity="major",
                        issue=f"Draft contains claims not supported by supplied evidence: {issue_names}.",
                        suggestion=(
                            "Revise these claims to match the provided code, innovation list, "
                            "and experiment tables, or add the missing evidence."
                        ),
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

    def _factual_consistency_checks(self, sections, experiments, innovations) -> list[dict[str, object]]:
        checks: list[dict[str, object]] = []
        section_values = sections.model_dump()
        evidence_text = self._evidence_text(experiments)
        checked_text = "\n".join(
            section_values.get(name, "")
            for name in ["abstract", "introduction", "experiments", "conclusion"]
        )

        checks.append(
            self._consistency_item(
                "unsupported_datasets",
                self._unsupported_datasets(checked_text, experiments),
            )
        )
        checks.append(
            self._consistency_item(
                "unsupported_metrics",
                self._unsupported_metrics(checked_text, experiments),
            )
        )
        checks.append(
            self._consistency_item(
                "unsupported_experiment_numbers",
                self._unsupported_numbers(checked_text, evidence_text),
            )
        )
        checks.append(
            self._consistency_item(
                "unsupported_method_threads",
                self._unsupported_method_threads(sections.method, innovations),
            )
        )
        return checks

    def _consistency_item(self, check: str, values: list[str]) -> dict[str, object]:
        return {
            "check": check,
            "status": "needs_review" if values else "ok",
            "values": values,
        }

    def _evidence_text(self, experiments) -> str:
        if not experiments:
            return ""
        return "\n".join(
            [
                experiments.raw_preview,
                *experiments.observations,
                *experiments.datasets,
                *experiments.metrics,
            ]
        )

    def _unsupported_datasets(self, text: str, experiments) -> list[str]:
        if not experiments or not experiments.datasets:
            return []
        allowed = {dataset.upper() for dataset in experiments.datasets}
        allowed.update(metric.upper() for metric in experiments.metrics)
        evidence_text = self._evidence_text(experiments)
        allowed.update(
            candidate.upper()
            for candidate in re.findall(r"\b[A-Z][A-Z0-9_]{1,8}(?:-[A-Z0-9_]{2,8})?\b", evidence_text)
            if candidate.upper() not in self.DATASET_STOPWORDS
        )
        candidates = set()
        for match in re.finditer(
            r"\b(?:across|over|using|datasets?|cohorts?|evaluat(?:e|ed|ing)\s+on|"
            r"train(?:ed|ing)?\s+on|test(?:ed|ing)?\s+on|validat(?:e|ed|ing)\s+on|"
            r"benchmark(?:ed|ing)?\s+on)\s+([^.;:\n]+)",
            text,
            flags=re.I,
        ):
            candidates.update(
                re.findall(r"\b[A-Z][A-Z0-9_]{1,8}(?:-[A-Z0-9_]{2,8})?\b", match.group(1))
            )
        return [
            candidate
            for candidate in sorted(candidates)
            if candidate.upper() not in allowed and candidate.upper() not in self.DATASET_STOPWORDS
        ]

    def _unsupported_metrics(self, text: str, experiments) -> list[str]:
        if not experiments or not experiments.metrics:
            return []
        allowed = {self._normalize_metric(metric) for metric in experiments.metrics}
        mentioned = {
            self._normalize_metric(match.group(1))
            for match in re.finditer(self.METRIC_PATTERN, text, flags=re.I)
        }
        return sorted(metric for metric in mentioned if metric and metric not in allowed)

    def _normalize_metric(self, metric: str) -> str:
        key = re.sub(r"\s+", " ", metric.strip().upper().replace("-", " "))
        return self.METRIC_ALIASES.get(key, key)

    def _unsupported_numbers(self, text: str, evidence_text: str) -> list[str]:
        if not evidence_text.strip():
            return []
        evidence_numbers = {
            self._normalize_number(match.group(0))
            for match in self._number_matches(evidence_text)
        }
        unsupported = []
        for match in self._number_matches(text):
            raw = match.group(0)
            if self._normalize_number(raw) not in evidence_numbers:
                unsupported.append(raw)
        return list(dict.fromkeys(unsupported))

    def _number_matches(self, text: str):
        return re.finditer(r"(?<![A-Za-z0-9])[-+]?(?:0?\.\d+|\d+\.\d+)(?:\s*%)?", text)

    def _normalize_number(self, raw: str) -> str:
        value = raw.strip().replace(" ", "")
        suffix = "%" if value.endswith("%") else ""
        value = value.rstrip("%")
        try:
            normalized = f"{float(value):.6g}"
        except ValueError:
            normalized = value.lstrip("+")
        return normalized + suffix

    def _unsupported_method_threads(self, method_text: str, innovations) -> list[str]:
        if not innovations:
            return []
        unsupported = []
        for heading, body in self._markdown_subsections(method_text):
            if self._generic_method_heading(heading):
                continue
            subsection = f"{heading}\n{body}"
            if not any(self._innovation_mentioned(subsection, innovation) for innovation in innovations):
                unsupported.append(heading)
        return unsupported

    def _markdown_subsections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_lines: list[str] = []

        def flush() -> None:
            if current_heading:
                sections.append((current_heading, "\n".join(current_lines).strip()))

        for line in text.splitlines():
            if line.startswith("### "):
                flush()
                current_heading = line[4:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        flush()
        return sections

    def _generic_method_heading(self, heading: str) -> bool:
        normalized = re.sub(r"[^a-z]+", " ", heading.lower()).strip()
        return normalized in {
            "overview",
            "method overview",
            "implementation details",
            "training details",
            "optimization",
        }

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
