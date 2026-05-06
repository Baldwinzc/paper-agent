"""Section writer."""

from __future__ import annotations

import json
import re
from typing import Any

from paper_agent.llm import ChatMessage, LLMClient, LLMError
from paper_agent.state import DraftSections, PaperState


class SectionWriterAgent:
    """Writes first-pass paper sections from the paper plan and innovation points."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def run(self, state: PaperState) -> PaperState:
        if self.llm_client and self.llm_client.available:
            try:
                state["sections"] = self._run_llm(state)
                state.setdefault("artifacts", {})["section_writer_mode"] = "llm"
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

        return DraftSections(
            abstract=(
                f"We study {request.project_name}, targeting {request.target_venue}. "
                f"Starting from the baseline paper, we identify a set of method-level "
                f"opportunities and propose an innovation-centered framework. "
                f"Our current draft centers on: {', '.join(item.name for item in innovations)}. "
                f"Preliminary experiments suggest the need for a structured evaluation over "
                f"{', '.join(experiments.datasets) if experiments and experiments.datasets else 'the target datasets'}."
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
                "First, discuss the direct baseline family and the assumptions inherited from it. "
                "Second, discuss methods related to each proposed innovation point. "
                "Third, clarify how the proposed work differs in motivation, mechanism, or evidence. "
                "Citation placeholders should be replaced after bibliography ingestion."
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
                f"Current missing details: {', '.join(experiments.missing_details) if experiments else 'experiment table needed'}."
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
        request = state["request"]
        baseline = state.get("baseline")
        code = state.get("code")
        experiments = state.get("experiments")
        innovations = state.get("innovations", [])
        outline = state.get("outline")
        venue = state.get("venue_template")

        prompt = {
            "task": "Draft core sections for a CS research paper.",
            "hard_rules": [
                "Write the Method section from innovation points, not raw code diffs.",
                "Use code and baseline only as evidence.",
                "Do not invent experiment numbers.",
                "If details are missing, write a precise placeholder instead of fabricating.",
                "Return strict JSON only.",
            ],
            "project_name": request.project_name,
            "target_venue": request.target_venue,
            "venue_template_family": venue.family if venue else "generic",
            "baseline": baseline.model_dump() if baseline else {},
            "code_summary": code.model_dump() if code else {},
            "experiment_summary": experiments.model_dump() if experiments else {},
            "innovations": [item.model_dump() for item in innovations],
            "outline": outline.model_dump() if outline else {},
            "required_json_schema": {
                "abstract": "string",
                "introduction": "string",
                "related_work": "string",
                "method": "string, use markdown ### headings for subsections",
                "experiments": "string, framework only if exact results are missing",
                "conclusion": "string",
            },
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
            max_tokens=5000,
            response_format={"type": "json_object"},
        )
        data = self._extract_json(result.content)
        return DraftSections(
            abstract=self._required_text(data, "abstract"),
            introduction=self._required_text(data, "introduction"),
            related_work=self._required_text(data, "related_work"),
            method=self._required_text(data, "method"),
            experiments=self._required_text(data, "experiments"),
            conclusion=self._required_text(data, "conclusion"),
        )

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
