"""Section writer."""

from __future__ import annotations

import json
import re
from typing import Any

from paper_agent.llm import ChatMessage, LLMClient, LLMError
from paper_agent.state import DraftSections, PaperState


class SectionWriterAgent:
    """Writes first-pass paper sections from the paper plan and innovation points."""

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

        innovation_text = "\n".join(
            f"- {item.name}: {item.technical_idea} Evidence: {'; '.join(item.evidence)}"
            for item in innovations
        )
        method_subsections = "\n\n".join(
            self._method_subsection(index, item) for index, item in enumerate(innovations, start=1)
        )
        citation_keys = state.setdefault("artifacts", {}).get("citation_keys", [])
        citation_hint = self._citation_hint(citation_keys)
        abstract_result_summary = self._result_summary(experiments, limit=1)
        experiment_result_summary = self._result_summary(experiments, limit=2)
        missing_details = (
            ", ".join(experiments.missing_details)
            if experiments and experiments.missing_details
            else "No missing experiment details detected by the analyzer"
        )

        return DraftSections(
            abstract=(
                f"We study {request.project_name}, targeting {request.target_venue}. "
                f"Starting from the baseline paper, we identify a set of method-level "
                f"opportunities and propose an innovation-centered framework. "
                f"Our current draft centers on: {', '.join(item.name for item in innovations)}. "
                f"{abstract_result_summary or 'Preliminary experiments suggest the need for a structured evaluation over ' + (', '.join(experiments.datasets) if experiments and experiments.datasets else 'the target datasets') + '.'}"
            ),
            introduction=(
                "The introduction should open with the research problem and its importance. "
                f"The baseline work, {baseline.title if baseline else 'the baseline paper'}, provides a strong "
                "starting point but leaves room for improvement. "
                f"Our central claim is: {outline.central_claim if outline else 'to be refined.'}\n\n"
                "The paper makes the following contributions:\n"
                f"{innovation_text}"
            ),
            related_work=(
                "Related work should be organized by research threads rather than as a flat list. "
                f"First, discuss the direct baseline family {citation_hint}. "
                "Second, discuss methods related to each proposed innovation point. "
                "Third, clarify how the proposed work differs in motivation, mechanism, or evidence. "
                "All generated citation metadata must be verified before submission."
            ),
            method=(
                "We describe the proposed method by following the innovation points established during "
                "analysis. Code-level details are used only as implementation evidence and are not treated "
                "as the paper narrative itself.\n\n"
                f"{method_subsections}"
            ),
            experiments=(
                "The experiments section should include: (1) datasets and preprocessing; "
                "(2) baseline methods; (3) evaluation metrics; (4) implementation details; "
                "(5) main comparison; (6) ablation studies; and (7) qualitative analysis. "
                f"{'Current parsed result summary: ' + experiment_result_summary + ' ' if experiment_result_summary else ''}"
                f"Current missing details: {missing_details}."
            ),
            conclusion=(
                "This paper presents an innovation-centered improvement over the baseline setting. "
                "The final conclusion should restate the validated contributions, summarize the main "
                "empirical findings, and honestly acknowledge limitations."
            ),
        )

    def _method_subsection(self, index, innovation) -> str:
        return (
            f"### {index}. {innovation.name}\n"
            f"Motivation. {innovation.motivation}\n\n"
            f"Design. {innovation.technical_idea}\n\n"
            f"Evidence. {'; '.join(innovation.evidence)}\n\n"
            f"Risk control. {innovation.risk}"
        )

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

        prompt = {
            "task": f"Draft the {section_name} section for a CS research paper.",
            "hard_rules": [
                "Write the Method section from innovation points, not raw code diffs.",
                "Use code and baseline only as evidence.",
                "Do not invent experiment numbers.",
                "If details are missing, write a precise placeholder instead of fabricating.",
                "Return only the requested section text.",
                "Do not wrap the answer in JSON or Markdown code fences.",
            ],
            "project_name": request.project_name,
            "target_venue": request.target_venue,
            "venue_template_family": venue.family if venue else "generic",
            "baseline": baseline.model_dump() if baseline else {},
            "code_summary": code.model_dump() if code else {},
            "experiment_summary": experiments.model_dump() if experiments else {},
            "innovations": [item.model_dump() for item in innovations],
            "bibliography": [entry.model_dump() for entry in bibliography],
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
        return self._clean_section_text(section_name, result.content)

    def _citation_hint(self, citation_keys: list[str]) -> str:
        if not citation_keys:
            return "with citation keys to be added"
        return "using seed citations such as " + ", ".join(
            rf"\cite{{{key}}}" for key in citation_keys[:3]
        )

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
