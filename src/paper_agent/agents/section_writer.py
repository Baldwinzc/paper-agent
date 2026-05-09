"""Section writer."""

from __future__ import annotations

import json
import re
from typing import Any

from paper_agent.llm import ChatMessage, LLMClient, LLMError
from paper_agent.state import DraftSections, PaperState


class SectionWriterAgent:
    """Writes first-pass paper sections from the paper plan and innovation points."""

    MISSING_RESULT_FORBIDDEN_PATTERNS = [
        ("C-index", r"\bC[-\s]?index\b|\bconcordance index\b"),
        ("IBS", r"\bIBS\b|\bintegrated brier\b"),
        ("AUC", r"\btime[-\s]?dependent AUC\b|\bAUC\b"),
        ("ablation", r"\bablation(?:s| studies)?\b"),
        ("statistical testing", r"\bWilcoxon\b|\bp[-\s]?value\b|\bstatistical significance\b"),
        ("cross-validation protocol", r"\bfive[-\s]?fold\b|\bcross[-\s]?validation\b"),
        ("percentage results", r"\b\d+(?:\.\d+)?\s*%"),
        ("empirical superiority", r"\boutperform(?:s|ed)?\b|\bimproves?\b|\bgains?\b|state-of-the-art|competitive"),
        ("completed evaluation", r"\bevaluated on\b|\bvalidated\b|\bbenchmarked\b"),
        ("final dataset claim", r"\bdataset used in this study comprises\b|\bthe complete dataset\b"),
    ]
    MISSING_RESULT_METHOD_FORBIDDEN_PATTERNS = [
        (
            "unsupported method outcome",
            r"\b(?:allows?|enables?)\s+(?:the\s+)?model\s+to\s+(?:capture|learn|improve|outperform)\b",
        ),
        (
            "unsupported mechanism effect",
            r"\bsubstantially\s+(?:simplif(?:y|ies|ying)|improv(?:e|es|ing)|enhanc(?:e|es|ing))\b",
        ),
    ]
    MISSING_RESULT_RELATED_WORK_FORBIDDEN_PATTERNS = [
        (
            "unsupported related-work method effect",
            r"\b(?:preserving|enabling|removing\s+the\s+need|suffices?|learns\s+soft|"
            r"unlike\s+prior|unlike\s+previous)\b",
        ),
    ]
    LLM_PROCEDURAL_FORBIDDEN_PATTERNS = [
        ("placeholder marker", r"\[(?:placeholder|todo|tbd)[^\]]*\]|\bPlaceholder:|\bTODO\b|\bTBD\b"),
        (
            "writer instruction",
            r"\b(?:should|must)\s+(?:be\s+)?(?:inserted|added|filled|completed|highlighted)\b",
        ),
        (
            "pending-result instruction",
            r"\bonce final (?:experimental )?results are available\b|\bthe table should\b",
        ),
    ]
    LLM_METHOD_DIFF_FORBIDDEN_PATTERNS = [
        (
            "baseline-diff framing",
            r"\b(?:baseline|prior work|previous method|existing method)\b.{0,90}"
            r"\b(?:replace|replaces|removed|remove|modify|modifies|differ|differs|instead|relative to)\b",
        ),
        (
            "baseline-diff framing",
            r"\b(?:replace|replaces|removed|remove|modify|modifies|differ|differs)\b.{0,90}"
            r"\b(?:baseline|prior work|previous method|existing method)\b",
        ),
        ("implementation-modification framing", r"\bmodifications affecting\b|\braw code diff\b|\bcode differences?\b"),
    ]
    EXPERIMENT_CLAIM_SECTIONS = {"abstract", "introduction", "experiments", "conclusion"}

    SECTION_SPECS = {
        "abstract": {
            "max_tokens": 450,
            "instruction": "Write one concise TPAMI-style abstract. Do not invent experiment numbers.",
        },
        "introduction": {
            "max_tokens": 900,
            "instruction": "Write the Introduction. Include motivation, baseline gap, and contributions.",
        },
        "related_work": {
            "max_tokens": 1100,
            "instruction": "Write Related Work organized by research threads. Use citation placeholders only when needed.",
        },
        "method": {
            "max_tokens": 1200,
            "instruction": (
                "Write Method as a standalone proposed design from innovation points only. "
                "Do not frame it as baseline/code modifications. Use markdown ### headings."
            ),
        },
        "experiments": {
            "max_tokens": 900,
            "instruction": "Write only an experiments framework. Since exact results may be missing, do not claim improvements or numeric deltas.",
        },
        "conclusion": {
            "max_tokens": 700,
            "instruction": "Write a cautious conclusion. Do not claim validation or gains without exact results.",
        },
    }

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def run(self, state: PaperState) -> PaperState:
        if self.llm_client and self.llm_client.available:
            try:
                state["sections"] = self._run_llm(state)
                state.setdefault("artifacts", {})["section_writer_mode"] = (
                    "partial_llm"
                    if state.setdefault("artifacts", {}).get("section_writer_section_errors")
                    else "llm"
                )
                return state
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                state.setdefault("artifacts", {})["section_writer_llm_error"] = str(exc)

        state["sections"] = self._run_fallback(state)
        state.setdefault("artifacts", {})["section_writer_mode"] = "fallback"
        return state

    def _run_fallback(self, state: PaperState) -> DraftSections:
        request = state["request"]
        baseline = state.get("baseline")
        experiments = state.get("experiments")
        innovations = state.get("innovations", [])
        outline = state.get("outline")

        method_subsections = "\n\n".join(
            self._method_subsection(index, item) for index, item in enumerate(innovations, start=1)
        )
        citation_keys = state.setdefault("artifacts", {}).get("citation_keys", [])
        citation_hint = self._citation_hint(citation_keys)
        related_work_discovery = state.get("artifacts", {}).get("related_work_candidates", [])
        related_work_text = self._related_work_text(
            related_work_discovery,
            citation_hint,
            baseline_title=baseline.title if baseline else "",
            innovations=innovations,
        )
        abstract_result_summary = self._result_summary(experiments, limit=1)
        experiment_result_summary = self._result_summary(experiments, limit=2)
        missing_details = experiments.missing_details if experiments else []
        abstract_evidence = self._abstract_evidence_text(experiments, abstract_result_summary)

        return DraftSections(
            abstract=(
                f"We study {request.project_name}, targeting {request.target_venue}. "
                f"Starting from the baseline paper, we identify a set of method-level "
                f"opportunities and propose an innovation-centered framework. "
                f"The proposed study centers on: {self._innovation_name_list(innovations)}. "
                f"{abstract_evidence}"
            ),
            introduction=self._introduction_text(request, baseline, outline, innovations),
            related_work=related_work_text,
            method=(
                "We describe the proposed method by following the innovation points established during "
                "analysis. Code-level details are used only as implementation evidence and are not treated "
                "as the paper narrative itself.\n\n"
                f"{method_subsections}"
            ),
            experiments=self._experiments_text(
                experiments,
                experiment_result_summary,
                missing_details,
            ),
            conclusion=self._conclusion_text(innovations, experiments, experiment_result_summary),
        )

    def _method_subsection(self, index, innovation) -> str:
        return (
            f"### {index}. {innovation.name}\n"
            f"Motivation. {self._paper_prose(innovation.motivation)}\n\n"
            f"Design. {self._paper_prose(innovation.technical_idea)}\n\n"
            f"Evidence. {self._evidence_text(innovation.evidence)}\n\n"
            f"Validation note. {self._risk_text(innovation.risk)}"
        )

    def _introduction_text(self, request, baseline, outline, innovations) -> str:
        baseline_title = baseline.title if baseline and baseline.title else "the supplied baseline"
        problem_sentence = self._problem_sentence(request, baseline)
        limitation_sentence = self._limitation_sentence(baseline)
        central_claim = (
            self._paper_prose(outline.central_claim)
            if outline and outline.central_claim
            else f"{request.project_name} can be improved through the proposed innovation set."
        )
        claim_text = (
            self._lower_initial(central_claim)
            if central_claim
            else "the proposed method addresses the identified gap."
        )

        return (
            f"{problem_sentence} The baseline work, {baseline_title}, establishes the immediate "
            f"starting point for this study. {limitation_sentence} "
            f"The central claim of this study is that {claim_text}\n\n"
            "The contributions are organized as follows:\n"
            f"{self._contribution_text(innovations)}"
        )

    def _experiments_text(
        self,
        experiments,
        result_summary: str,
        missing_details: list[str],
    ) -> str:
        datasets = self._list_phrase(
            experiments.datasets if experiments else [],
            "the datasets supplied by the author",
        )
        metrics = self._list_phrase(
            experiments.metrics if experiments else [],
            "the evaluation metrics supplied by the author",
        )
        missing = [self._paper_prose(item).rstrip(".") for item in missing_details]
        result_tables = experiments.result_tables if experiments else []
        ablation_evidence = experiments.ablation_evidence if experiments else []

        if result_tables:
            main_results = (
                "### Main Results\n"
                + self._structured_main_results_text(result_tables)
            )
        elif result_summary:
            main_results = (
                "### Main Results\n"
                f"The parsed result tables report the following evidence: {result_summary} "
                "These statements come directly from the supplied experiment file and remain bounded "
                "to the reported tables."
            )
        else:
            main_results = (
                "### Main Results\n"
                "The supplied materials do not yet contain a structured numeric result table. The draft "
                "therefore records the evaluation protocol and leaves performance claims for the final "
                "author-verified results."
            )

        if missing:
            completion = (
                "### Completion Items\n"
                "The remaining experiment details are "
                + "; ".join(missing)
                + ". These items mark the places where the author must add concrete protocol or result "
                "information before finalizing empirical conclusions."
            )
        else:
            completion = (
                "### Completion Items\n"
                "The analyzer found explicit dataset names, metric names, and baseline comparison rows "
                "in the supplied notes. Implementation settings, statistical testing, and final table "
                "formatting still require manual verification before submission."
            )

        if missing:
            setup = (
                "### Experimental Setup\n"
                f"The planned evaluation section is organized around {datasets}. The supplied materials currently "
                "support dataset and cohort description only; metric definitions, comparison rows, training "
                "settings, and additional analyses should be added after real runs are available."
            )
        else:
            setup = (
                "### Experimental Setup\n"
                f"The evaluation is organized around {datasets} and reports {metrics}. The comparison "
                "centers on the provided baseline family and the proposed method, with additional analyses "
                "added when the supplied evidence supports them."
            )

        sections = [setup, main_results]
        if ablation_evidence:
            sections.append(self._ablation_evidence_text(ablation_evidence))
        sections.append(completion)
        return "\n\n".join(sections)

    def _structured_main_results_text(self, result_tables) -> str:
        table_sentences = []
        example_sentences = []
        for table in result_tables[:3]:
            comparisons = table.comparisons
            if not comparisons:
                continue
            wins = sum(1 for comparison in comparisons if comparison.improved)
            average_delta = sum(comparison.signed_improvement for comparison in comparisons) / len(comparisons)
            metric = table.metric or "reported metrics"
            table_sentences.append(
                f"In {table.caption}, {table.method} improves over {table.baseline} on "
                f"{wins}/{len(comparisons)} {metric} comparisons, with an average signed "
                f"improvement of {average_delta:+.3f}."
            )
            strongest = sorted(
                comparisons,
                key=lambda item: abs(item.signed_improvement),
                reverse=True,
            )[:2]
            for comparison in strongest:
                example_sentences.append(self._comparison_sentence(comparison))
        body = " ".join(table_sentences)
        if example_sentences:
            body += " Representative comparisons include " + "; ".join(example_sentences[:4]) + "."
        return (
            body
            + " These claims are copied from the supplied experiment tables and should not be "
            "extended beyond the reported metrics."
        )

    def _comparison_sentence(self, comparison) -> str:
        metric = f" {comparison.metric}" if comparison.metric else ""
        dataset = comparison.dataset or "the reported column"
        direction = "higher" if comparison.higher_is_better else "lower"
        return (
            f"{dataset}{metric}: {comparison.method_value:.3f} vs "
            f"{comparison.baseline_value:.3f} ({comparison.signed_improvement:+.3f}, "
            f"{direction} is better)"
        )

    def _ablation_evidence_text(self, ablation_evidence) -> str:
        selected = sorted(
            ablation_evidence,
            key=lambda item: abs(item.signed_drop),
            reverse=True,
        )[:5]
        sentences = []
        for item in selected:
            metric = f" {item.metric}" if item.metric else ""
            dataset = item.dataset or "the reported column"
            direction = "higher" if item.higher_is_better else "lower"
            support = (
                "; supports " + ", ".join(item.supports[:2])
                if item.supports
                else ""
            )
            sentences.append(
                f"{item.variant} changes {dataset}{metric} from "
                f"{item.reference_value:.3f} to {item.variant_value:.3f} "
                f"(signed drop {item.signed_drop:+.3f}, {direction} is better{support})"
            )
        return (
            "### Ablation Evidence\n"
            + "; ".join(sentences)
            + ". These component-level statements are limited to the supplied ablation tables."
        )

    def _conclusion_text(self, innovations, experiments, result_summary: str) -> str:
        innovation_names = self._innovation_name_list(innovations)
        has_missing_empirical_details = bool(experiments and experiments.missing_details)
        if result_summary:
            evidence_sentence = (
                f"The supplied experiment tables provide preliminary evidence summarized as: {result_summary}"
            )
        elif experiments and experiments.missing_details:
            evidence_sentence = (
                "The empirical section remains incomplete because several protocol or result details "
                "are still missing from the supplied materials."
            )
        else:
            evidence_sentence = (
                "The empirical discussion remains intentionally cautious until the final verified result "
                "tables are inserted."
            )
        contribution_frame = (
            "as the main contribution set for studying the baseline setting"
            if has_missing_empirical_details
            else "as the main contribution set for improving the baseline setting"
        )

        return (
            f"This paper frames {innovation_names} {contribution_frame}. "
            f"The study separates the proposed technical ideas from the code-level "
            f"evidence used to support them, which keeps the method narrative centered on the analyzed "
            f"innovations. {evidence_sentence} Before submission, the author still needs to verify "
            f"bibliography metadata, final experiment details, and the strength of each novelty claim."
        )

    def _abstract_evidence_text(self, experiments, result_summary: str) -> str:
        if result_summary:
            return result_summary
        datasets = (
            ", ".join(experiments.datasets)
            if experiments and experiments.datasets
            else "the target datasets"
        )
        return (
            f"Available cohort metadata supports a structured evaluation plan over {datasets}; "
            "performance evaluation remains pending."
        )

    def _problem_sentence(self, request, baseline) -> str:
        if baseline and baseline.problem:
            return self._paper_prose(baseline.problem).rstrip(".") + "."
        return (
            f"{request.project_name} focuses on a research problem targeted at "
            f"{request.target_venue}, where a baseline method is extended through the supplied code "
            "and experiment evidence."
        )

    def _limitation_sentence(self, baseline) -> str:
        if baseline and baseline.limitations:
            limitation = self._paper_prose(baseline.limitations[0]).rstrip(".")
            return f"The key limitation carried into this draft is: {limitation}."
        return (
            "The limitation analysis is conservative because it is derived only from the provided "
            "baseline, code, and experiment artifacts."
        )

    def _contribution_text(self, innovations) -> str:
        if not innovations:
            return (
                "- The draft identifies the baseline setting and reserves the technical "
                "contribution for author-supplied method notes."
            )
        bullets = []
        for innovation in innovations:
            name = self._paper_prose(innovation.name)
            idea = self._paper_prose(innovation.technical_idea).rstrip(".")
            evidence = self._evidence_text(innovation.evidence)
            bullets.append(f"- {name}. {idea}. Supporting evidence: {evidence}")
        return "\n".join(bullets)

    def _evidence_text(self, evidence: list[str]) -> str:
        cleaned = [self._paper_prose(item).rstrip(".") for item in evidence if item.strip()]
        if not cleaned:
            return "Evidence remains to be supplied."
        selected: list[str] = []
        seen_labels: set[str] = set()
        generic: list[str] = []
        for item in cleaned:
            if item.lower().startswith("scanned "):
                generic.append(item)
                continue
            label_match = re.search(r"\(([^)]+)\)", item)
            label = label_match.group(1).lower() if label_match else item[:60].lower()
            if label in seen_labels:
                continue
            selected.append(item)
            seen_labels.add(label)
            if len(selected) >= 5:
                break
        if len(selected) < 3:
            selected.extend(item for item in generic if item not in selected)
        return "; ".join(selected[:5]) + "."

    def _risk_text(self, risk: str) -> str:
        text = self._paper_prose(risk).strip()
        if not text:
            return "Novelty, wording, and empirical support require manual verification before submission."
        return text

    def _innovation_name_list(self, innovations) -> str:
        if not innovations:
            return "a targeted method contribution to be finalized by the author"
        return ", ".join(item.name for item in innovations[:4])

    def _list_phrase(self, values: list[str], fallback: str) -> str:
        cleaned = [value for value in values if value]
        if not cleaned:
            return fallback
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return " and ".join(cleaned)
        return ", ".join(cleaned[:-1]) + ", and " + cleaned[-1]

    def _lower_initial(self, text: str) -> str:
        return text[:1].lower() + text[1:] if text else text

    def _paper_prose(self, text: str) -> str:
        replacements = {
            r"\bUser should provide the core technical change in method notes\.?": (
                "The core technical change remains to be specified in the method notes."
            ),
            r"\buser should confirm novelty and wording\.?": (
                "novelty and wording require manual confirmation."
            ),
            r"\bthis point should not be treated as final\.?": "this point remains provisional.",
            r"\bBaseline comparison rows should be made explicit\.?": (
                "Baseline comparison rows are not yet explicit."
            ),
            r"\bDataset names are not explicit\.?": "Dataset names are not yet explicit.",
            r"\bEvaluation metrics are not explicit\.?": "Evaluation metrics are not yet explicit.",
        }
        cleaned = text.strip()
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _run_llm(self, state: PaperState) -> DraftSections:
        fallback = self._run_fallback(state)
        values: dict[str, str] = {}
        errors: dict[str, str] = {}
        successes: list[str] = []
        for section_name in self.SECTION_SPECS:
            try:
                values[section_name] = self._run_llm_section(state, section_name)
                successes.append(section_name)
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                errors[section_name] = str(exc)
                values[section_name] = getattr(fallback, section_name)

        state.setdefault("artifacts", {})["section_writer_llm_attempted_sections"] = list(
            self.SECTION_SPECS
        )
        state["artifacts"]["section_writer_llm_successes"] = successes
        if errors:
            state["artifacts"]["section_writer_section_errors"] = errors

        return DraftSections(**values)

    def _run_llm_section(self, state: PaperState, section_name: str) -> str:
        request = state["request"]
        baseline = state.get("baseline")
        code = state.get("code")
        experiments = state.get("experiments")
        innovations = state.get("innovations", [])
        outline = state.get("outline")
        venue = state.get("venue_template")
        bibliography = state.get("bibliography", [])
        spec = self.SECTION_SPECS[section_name]
        missing_experiment_details = experiments.missing_details if experiments else []

        prompt = {
            "task": f"Draft the {section_name} section for a CS research paper.",
            "hard_rules": self._llm_hard_rules(missing_experiment_details),
            "project_name": request.project_name,
            "target_venue": request.target_venue,
            "venue_template_family": venue.family if venue else "generic",
            "baseline": baseline.model_dump() if baseline else {},
            "code_summary": code.model_dump() if code else {},
            "experiment_summary": experiments.model_dump() if experiments else {},
            "missing_experiment_details": missing_experiment_details,
            "innovations": [item.model_dump() for item in innovations],
            "bibliography": [entry.model_dump() for entry in bibliography],
            "related_work_discovery": state.get("artifacts", {}).get("related_work_candidates", []),
            "allowed_citation_keys": [entry.key for entry in bibliography],
            "outline": outline.model_dump() if outline else {},
            "section_name": section_name,
            "section_instruction": spec["instruction"],
            "output_format": f"Plain text content for {section_name}.",
        }

        result = self.llm_client.chat(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You are a careful academic writing agent for computer science papers. "
                        "You help draft paper sections from supplied evidence. You avoid overclaiming."
                    ),
                ),
                ChatMessage(role="user", content=json.dumps(prompt, ensure_ascii=False)),
            ],
            temperature=0.25,
            max_tokens=spec["max_tokens"],
        )
        section_text = self._clean_section_text(section_name, result.content)
        self._raise_if_missing_results_overclaimed(
            section_text, section_name, missing_experiment_details
        )
        self._raise_if_procedural_language(section_text, section_name)
        self._raise_if_method_diff_framing(section_text, section_name)
        self._raise_if_unsupported_experiment_claims(
            section_text,
            section_name,
            experiments,
        )
        return section_text

    def _raise_if_missing_results_overclaimed(
        self,
        section_text: str,
        section_name: str,
        missing_experiment_details: list[str],
    ) -> None:
        if not missing_experiment_details:
            return
        matched = [
            label
            for label, pattern in self.MISSING_RESULT_FORBIDDEN_PATTERNS
            if re.search(pattern, section_text, flags=re.I)
        ]
        if section_name == "method":
            matched.extend(
                label
                for label, pattern in self.MISSING_RESULT_METHOD_FORBIDDEN_PATTERNS
                if re.search(pattern, section_text, flags=re.I)
            )
        if section_name == "related_work":
            matched.extend(
                label
                for label, pattern in self.MISSING_RESULT_RELATED_WORK_FORBIDDEN_PATTERNS
                if re.search(pattern, section_text, flags=re.I)
            )
        if not matched:
            return
        raise ValueError(
            f"LLM {section_name} section included unsupported empirical language while "
            f"experiment details are missing: {', '.join(matched)}"
        )

    def _raise_if_procedural_language(self, section_text: str, section_name: str) -> None:
        matched = [
            label
            for label, pattern in self.LLM_PROCEDURAL_FORBIDDEN_PATTERNS
            if re.search(pattern, section_text, flags=re.I)
        ]
        if not matched:
            return
        raise ValueError(
            f"LLM {section_name} section included draft instructions or placeholders: "
            f"{', '.join(matched)}"
        )

    def _raise_if_unsupported_experiment_claims(
        self,
        section_text: str,
        section_name: str,
        experiments,
    ) -> None:
        if section_name not in self.EXPERIMENT_CLAIM_SECTIONS or not experiments:
            return

        from paper_agent.agents.reviewer import ReviewerAgent

        reviewer = ReviewerAgent()
        evidence_text = reviewer._evidence_text(experiments)
        issues = {
            "datasets": reviewer._unsupported_datasets(section_text, experiments),
            "metrics": reviewer._unsupported_metrics(section_text, experiments),
            "numbers": reviewer._unsupported_numbers(section_text, evidence_text),
        }
        issues = {key: values for key, values in issues.items() if values}
        if not issues:
            return
        details = "; ".join(
            f"{key}: {', '.join(values[:5])}"
            for key, values in issues.items()
        )
        raise ValueError(
            f"LLM {section_name} section included unsupported experiment claims: {details}"
        )

    def _llm_hard_rules(self, missing_experiment_details: list[str]) -> list[str]:
        rules = [
            "Write the Method section from innovation points, not raw code diffs.",
            "Use code and baseline only as evidence.",
            "For Method, do not describe the design as baseline modifications, code differences, "
            "or replacement of baseline components. Present the proposed computation as a standalone method.",
            "Do not invent experiment numbers.",
            "Do not invent preprocessing accuracies, classifier accuracies, hidden validation "
            "scores, hardware details, optimizer settings, or dataset properties absent from "
            "the supplied evidence.",
            "If details are missing, write cautious paper prose that says the evidence is pending "
            "instead of fabricating details.",
            "Do not include writer instructions, TODOs, TBDs, bracketed placeholders, or text "
            "telling the author what a future table should contain.",
            "For Related Work, write actual comparative paragraphs from "
            "related_work_discovery rather than instructions.",
            "Use only allowed_citation_keys for citations. Do not copy numeric citations like [5] "
            "or [18, 19] from baseline text.",
            "Return only the requested section text.",
            "Do not wrap the answer in JSON or Markdown code fences.",
        ]
        if missing_experiment_details:
            rules.extend(
                [
                    "The experiment evidence is incomplete. Do not claim that the method is evaluated, "
                    "validated, compared, competitive, state-of-the-art, or effective.",
                    "Do not mention C-index, IBS, ablation, sensitivity, p-values, results, gains, "
                    "or performance unless those exact values are present in the supplied experiment data.",
                    "When writing Abstract, Introduction, Experiments, or Conclusion, describe only the "
                    "available dataset/cohort summary and mark performance evaluation as pending.",
                    "When writing Method, describe intended computation and implementation evidence only; "
                    "avoid outcome language such as allows the model to capture, enables improvement, "
                    "or substantially improves unless supported by supplied ablation evidence.",
                    "When writing Related Work, focus on prior work and cautious positioning; do not claim "
                    "the proposed method preserves structure, removes the need for prior mechanisms, learns "
                    "better weights, or differs from all prior work without explicit evidence.",
                ]
            )
        return rules

    def _raise_if_method_diff_framing(self, section_text: str, section_name: str) -> None:
        if section_name != "method":
            return
        matched = [
            label
            for label, pattern in self.LLM_METHOD_DIFF_FORBIDDEN_PATTERNS
            if re.search(pattern, section_text, flags=re.I | re.S)
        ]
        if not matched:
            return
        raise ValueError(
            "LLM method section framed the design as code/baseline differences: "
            + ", ".join(sorted(set(matched)))
        )

    def _citation_hint(self, citation_keys: list[str]) -> str:
        if not citation_keys:
            return "with citation keys to be added"
        return "using seed citations such as " + ", ".join(
            rf"\cite{{{key}}}" for key in citation_keys[:3]
        )

    def _related_work_text(
        self,
        candidates: list[dict[str, Any]],
        citation_hint: str,
        baseline_title: str,
        innovations,
    ) -> str:
        if not candidates:
            return (
                "### Baseline Family and Problem Setting\n"
                f"The direct baseline family provides the starting context {citation_hint}. "
                "This thread explains the task definition, common evaluation protocols, "
                "and the specific limitations that motivate the proposed method.\n\n"
                "### Method Threads Related to the Proposed Contributions\n"
                "The remaining related work is organized around the accepted innovation points. "
                "Each thread connects prior modeling choices to one proposed contribution, while "
                "avoiding unsupported novelty claims until the bibliography is manually verified."
            )

        grouped: dict[str, list[dict[str, Any]]] = self._group_related_work(candidates)
        sections = []
        if grouped.get("baseline_reference") or grouped.get("baseline_citing") or grouped.get("baseline_mentioned"):
            baseline_items = grouped.get("baseline_reference", [])[:3]
            citing_items = grouped.get("baseline_citing", [])[:3]
            mentioned_items = grouped.get("baseline_mentioned", [])[:3]
            pieces = []
            if baseline_title:
                pieces.append(
                    f"The provided baseline, {baseline_title}, anchors the local problem setting."
                )
            if mentioned_items:
                pieces.append(
                    "Candidate works retrieved from names explicitly discussed by the baseline include "
                    + self._candidate_sentence(mentioned_items)
                    + ", which provides local context for the baseline's stated research threads after manual verification."
                )
            if baseline_items:
                pieces.append(
                    "Its cited lineage includes "
                    + self._candidate_sentence(baseline_items)
                    + ", which helps identify the assumptions inherited by the baseline."
                )
            if citing_items:
                pieces.append(
                    "Later papers that cite the baseline include "
                    + self._candidate_sentence(citing_items)
                    + ", which indicate how the baseline has been extended or repositioned."
                )
            sections.append("### Baseline Lineage\n" + " ".join(pieces))

        if grouped.get("influential"):
            sections.append(
                "### Influential Field Context\n"
                "High-impact work in the broader field includes "
                + self._candidate_sentence(grouped["influential"][:3])
                + ". These papers provide the main methodological context for positioning the "
                "proposed approach against established whole-slide survival prediction and "
                "computational pathology models."
            )

        if grouped.get("recent"):
            sections.append(
                "### Recent Developments\n"
                "Recent work includes "
                + self._candidate_sentence(grouped["recent"][:3])
                + ". This thread is useful for contrasting the proposed method with newer "
                "foundation-model, "
                "multimodal, or structure-aware survival prediction systems."
            )

        innovation_names = (
            ", ".join(item.name for item in innovations[:3])
            if innovations
            else "the proposed contributions"
        )
        sections.append(
            "### Relation to the Proposed Method\n"
            f"The proposed method is positioned around {innovation_names}. "
            "This framing separates prior assumptions inherited from the baseline family, "
            "mechanisms changed by the proposed design, and comparisons that still require "
            "verified bibliography metadata before submission."
        )
        return "\n\n".join(sections)

    def _group_related_work(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            grouped.setdefault(str(candidate.get("category", "unknown")), []).append(candidate)
        return grouped

    def _candidate_sentence(self, items: list[dict[str, Any]]) -> str:
        phrases = [self._candidate_phrase(item) for item in items if item.get("key")]
        if not phrases:
            return "the discovered candidate papers"
        if len(phrases) == 1:
            return phrases[0]
        return ", ".join(phrases[:-1]) + ", and " + phrases[-1]

    def _candidate_phrase(self, item: dict[str, Any]) -> str:
        title = str(item.get("title") or "a related paper")
        year = item.get("year")
        cited_by = item.get("cited_by_count", 0)
        key = item.get("key")
        context = []
        if year:
            context.append(str(year))
        if cited_by:
            context.append(f"cited by {cited_by}")
        suffix = f" ({', '.join(context)})" if context else ""
        return f"{title}{suffix} " + rf"\cite{{{key}}}"

    def _result_summary(self, experiments, limit: int) -> str:
        if not experiments:
            return ""
        useful = [
            observation
            for observation in experiments.observations
            if observation != "Experiment analysis needs more structured result tables."
        ]
        return " ".join(useful[:limit])

    def _extract_json(self, content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                raise ValueError("LLM response is not JSON.")
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("LLM JSON response must be an object.")
        return data

    def _required_text(self, data: dict[str, Any], key: str) -> str:
        value = data.get(key, "")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"LLM JSON missing non-empty {key}.")
        return value.strip()

    def _clean_section_text(self, section_name: str, content: str) -> str:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:\w+)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        text = self._drop_repeated_section_heading(section_name, text)
        text = self._remove_numeric_citations(text)
        if not text:
            raise ValueError("LLM section response is empty.")
        return text.strip()

    def _remove_numeric_citations(self, text: str) -> str:
        text = re.sub(r"\s*\[(?:\d+\s*(?:,\s*\d+\s*)*)\]", "", text)
        return re.sub(r"\s+([,.;:])", r"\1", text)

    def _drop_repeated_section_heading(self, section_name: str, text: str) -> str:
        lines = text.splitlines()
        while lines and self._is_repeated_heading(section_name, lines[0]):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
        return "\n".join(lines)

    def _is_repeated_heading(self, section_name: str, line: str) -> bool:
        normalized = re.sub(r"[^a-z]+", " ", line.lower()).strip()
        normalized = re.sub(r"^\d+\s+", "", normalized).strip()
        section_aliases = {
            "abstract": {"abstract"},
            "introduction": {"introduction"},
            "related_work": {"related work", "related works"},
            "method": {"method", "methods"},
            "experiments": {"experiments", "experiment"},
            "conclusion": {"conclusion", "conclusions"},
        }
        return normalized in section_aliases.get(section_name, {section_name})
