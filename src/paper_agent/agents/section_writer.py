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
    ]

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
            "instruction": "Write Method from innovation points only. Use markdown ### headings for subsections.",
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

        if result_summary:
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
                f"The empirical section is organized around {datasets}. The supplied materials currently "
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

        return f"{setup}\n\n{main_results}\n\n{completion}"

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
        return f"Preliminary experiments motivate a structured evaluation over {datasets}."

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
        return "; ".join(cleaned[:3]) + "."

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
        for section_name in self.SECTION_SPECS:
            try:
                values[section_name] = self._run_llm_section(state, section_name)
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                errors[section_name] = str(exc)
                values[section_name] = getattr(fallback, section_name)

        if errors:
            state.setdefault("artifacts", {})["section_writer_section_errors"] = errors

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
        if not matched:
            return
        raise ValueError(
            f"LLM {section_name} section included unsupported empirical language while "
            f"experiment details are missing: {', '.join(matched)}"
        )

    def _llm_hard_rules(self, missing_experiment_details: list[str]) -> list[str]:
        rules = [
            "Write the Method section from innovation points, not raw code diffs.",
            "Use code and baseline only as evidence.",
            "Do not invent experiment numbers.",
            "If details are missing, write a precise placeholder instead of fabricating.",
            "For Related Work, write actual comparative paragraphs from "
            "related_work_discovery rather than instructions.",
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
                ]
            )
        return rules

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
        if grouped.get("baseline_reference") or grouped.get("baseline_citing"):
            baseline_items = grouped.get("baseline_reference", [])[:3]
            citing_items = grouped.get("baseline_citing", [])[:3]
            pieces = []
            if baseline_title:
                pieces.append(
                    f"The provided baseline, {baseline_title}, anchors the local problem setting."
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
        if not text:
            raise ValueError("LLM section response is empty.")
        return text.strip()

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
