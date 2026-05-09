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
        "II",
        "III",
        "IV",
        "IX",
        "L_",
        "L_REC",
        "LLM",
        "OT",
        "OTSU",
        "PDF",
        "TCGA",
        "TME",
        "TPAMI",
        "UNI",
        "VI",
        "VII",
        "VIII",
        "WSI",
        "WSIS",
        "XI",
        "XII",
    }
    STRONG_ABLATION_TOKENS = {
        "attention",
        "bidirectional",
        "cross-attention",
        "hcon",
        "l_rec",
        "reconstruction",
        "wasserstein",
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
            traceability = self._innovation_traceability(sections.method, innovations, experiments)
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
            consistency = self._factual_consistency_checks(
                sections,
                experiments,
                innovations,
                state.get("code"),
                state.get("artifacts", {}),
            )
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

    def _factual_consistency_checks(
        self,
        sections,
        experiments,
        innovations,
        code=None,
        artifacts=None,
    ) -> list[dict[str, object]]:
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
                self._unsupported_method_threads(
                    sections.method,
                    innovations,
                    experiments,
                    code,
                    artifacts or {},
                ),
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
        result_values = []
        for table in experiments.result_tables:
            for comparison in table.comparisons:
                result_values.extend(
                    [
                        f"{comparison.method_value:.3f}",
                        f"{comparison.baseline_value:.3f}",
                        f"{comparison.signed_improvement:+.3f}",
                    ]
                )
        for ablation in experiments.ablation_evidence:
            result_values.extend(
                [
                    f"{ablation.reference_value:.3f}",
                    f"{ablation.variant_value:.3f}",
                    f"{ablation.signed_drop:+.3f}",
                    ablation.reference,
                    ablation.variant,
                    *ablation.supports,
                ]
            )
        for sensitivity in experiments.sensitivity_evidence:
            result_values.extend(
                [
                    sensitivity.parameter,
                    sensitivity.best_parameter_value,
                    f"{sensitivity.best_metric_value:.3f}",
                    f"{sensitivity.worst_metric_value:.3f}",
                    *sensitivity.tested_values,
                    *(f"{value:.3f}" for value in sensitivity.metric_values),
                ]
            )
        for test in experiments.statistical_tests:
            result_values.extend(
                [
                    test.comparison,
                    test.metric,
                    test.test,
                    test.p_value_text,
                    f"{test.p_value:.3f}",
                    f"{test.alpha:.2f}",
                ]
            )
        return "\n".join(
            [
                experiments.raw_preview,
                *experiments.observations,
                *experiments.datasets,
                *experiments.metrics,
                *result_values,
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
        if "IBS" in allowed:
            allowed.add("BRIER SCORE")
        if "BRIER SCORE" in allowed:
            allowed.add("IBS")
        mentioned = set()
        for match in re.finditer(self.METRIC_PATTERN, text, flags=re.I):
            metric = self._normalize_metric(match.group(1))
            if metric == "ACCURACY" and not self._metric_claim_has_number(text, match):
                continue
            mentioned.add(metric)
        return sorted(metric for metric in mentioned if metric and metric not in allowed)

    def _metric_claim_has_number(self, text: str, match: re.Match[str]) -> bool:
        window = text[max(0, match.start() - 80) : match.end() + 80]
        return bool(re.search(r"\d", window))

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

    def _unsupported_method_threads(
        self,
        method_text: str,
        innovations,
        experiments=None,
        code=None,
        artifacts=None,
    ) -> list[str]:
        if not innovations:
            return []
        unsupported = []
        for heading, body in self._markdown_subsections(method_text):
            if self._generic_method_heading(heading):
                continue
            subsection = f"{heading}\n{body}"
            innovation_supported = any(
                self._innovation_mentioned(subsection, innovation)
                for innovation in innovations
            )
            ablation_supported = self._method_thread_supported_by_ablation(subsection, experiments)
            implementation_supported = self._method_thread_supported_by_implementation(
                subsection,
                code,
                artifacts or {},
            )
            if not innovation_supported and not ablation_supported and not implementation_supported:
                unsupported.append(heading)
        return unsupported

    def _method_thread_supported_by_ablation(self, subsection: str, experiments) -> bool:
        if not experiments or not experiments.ablation_evidence:
            return False
        subsection_tokens = self._expanded_content_tokens(subsection)
        for item in experiments.ablation_evidence:
            evidence_tokens = self._expanded_content_tokens(" ".join([item.variant, *item.supports]))
            overlap = subsection_tokens & evidence_tokens
            if (overlap & self.STRONG_ABLATION_TOKENS) or len(overlap) >= 2:
                return True
        return False

    def _method_thread_supported_by_implementation(
        self,
        subsection: str,
        code,
        artifacts: dict[str, object],
    ) -> bool:
        evidence_items = self._implementation_support_items(code, artifacts)
        if not evidence_items:
            return False
        subsection_tokens = self._expanded_content_tokens(subsection)
        if not subsection_tokens:
            return False
        for item in evidence_items:
            evidence_tokens = self._expanded_content_tokens(item)
            overlap = subsection_tokens & evidence_tokens
            if (overlap & self._strong_method_tokens()) or len(overlap) >= 3:
                return True
        return False

    def _implementation_support_items(self, code, artifacts: dict[str, object]) -> list[str]:
        items: list[str] = []
        if code:
            items.extend(getattr(code, "likely_method_files", []) or [])
            items.extend(getattr(code, "implementation_evidence", []) or [])
            items.extend(getattr(code, "method_claims", []) or [])
            if getattr(code, "summary", ""):
                items.append(code.summary)

        comparison = artifacts.get("code_baseline_comparison", {})
        if isinstance(comparison, dict):
            items.extend(str(term) for term in comparison.get("overlapping_terms", []) or [])
            items.extend(str(term) for term in comparison.get("code_only_terms", []) or [])
            items.extend(str(seed) for seed in comparison.get("innovation_seeds", []) or [])
            for shift in comparison.get("likely_method_shifts", []) or []:
                if not isinstance(shift, dict):
                    continue
                items.append(str(shift.get("technique", "")))
                items.append(str(shift.get("rationale", "")))
                items.extend(str(evidence) for evidence in shift.get("evidence", []) or [])

        return [item for item in items if item.strip()]

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

    def _innovation_traceability(self, method_text: str, innovations, experiments=None) -> list[dict[str, object]]:
        ablation_evidence = experiments.ablation_evidence if experiments else []
        traceability = []
        for innovation in innovations:
            matches = self._matching_ablation_evidence(innovation, ablation_evidence)
            traceability.append(
                {
                    "name": innovation.name,
                    "mentioned_in_method": self._innovation_mentioned(method_text, innovation),
                    "evidence_count": len(innovation.evidence),
                    "evidence_preview": innovation.evidence[:2],
                    "ablation_evidence_count": len(matches),
                    "ablation_evidence_preview": [
                        self._format_ablation_evidence(item) for item in matches[:2]
                    ],
                }
            )
        return traceability

    def _matching_ablation_evidence(self, innovation, ablation_evidence) -> list:
        innovation_text = f"{innovation.name} {innovation.technical_idea}"
        innovation_tokens = self._expanded_content_tokens(innovation_text)
        if not innovation_tokens:
            return []
        matches = []
        for item in ablation_evidence:
            variant_tokens = self._expanded_content_tokens(item.variant)
            support_tokens = self._expanded_content_tokens(" ".join(item.supports))
            overlap = innovation_tokens & (variant_tokens | support_tokens)
            strong_overlap = overlap & self.STRONG_ABLATION_TOKENS
            phrase_match = any(
                self._ablation_support_matches(support, innovation_text, innovation_tokens)
                for support in item.supports
            )
            if phrase_match or strong_overlap or len(overlap) >= 2:
                matches.append(item)
        return matches

    def _ablation_support_matches(
        self,
        support: str,
        innovation_text: str,
        innovation_tokens: set[str],
    ) -> bool:
        normalized_support = support.lower()
        normalized_innovation = innovation_text.lower()
        if normalized_support in normalized_innovation:
            return True
        support_tokens = self._expanded_content_tokens(support)
        if len(support_tokens & innovation_tokens) >= 2:
            return True
        return bool((support_tokens & innovation_tokens) & self.STRONG_ABLATION_TOKENS)

    def _expanded_content_tokens(self, text: str) -> set[str]:
        expanded = re.sub(r"[-_]+", " ", text)
        tokens = set(self._content_tokens(text)) | set(self._content_tokens(expanded))
        singulars = {
            token[:-1]
            for token in tokens
            if token.endswith("s") and len(token) > 4
        }
        return tokens | singulars

    def _strong_method_tokens(self) -> set[str]:
        return {
            "attention",
            "barycenter",
            "bidirectional",
            "binary",
            "convolution",
            "cox",
            "cross",
            "fusion",
            "hcon",
            "hyperedge",
            "hypergraph",
            "incidence",
            "optimal",
            "prototype",
            "reconstruction",
            "survival",
            "transport",
            "wasserstein",
        }

    def _format_ablation_evidence(self, item) -> str:
        metric = f" {item.metric}" if item.metric else ""
        dataset = item.dataset or "reported column"
        return (
            f"{item.variant} on {dataset}{metric}: {item.reference_value:.3f} -> "
            f"{item.variant_value:.3f} (signed drop {item.signed_drop:+.3f})"
        )

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
