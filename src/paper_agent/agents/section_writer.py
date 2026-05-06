"""Section writer."""

from __future__ import annotations

from paper_agent.state import DraftSections, PaperState


class SectionWriterAgent:
    """Writes first-pass paper sections from the paper plan and innovation points."""

    def run(self, state: PaperState) -> PaperState:
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

        state["sections"] = DraftSections(
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
        return state

    def _method_subsection(self, index, innovation) -> str:
        return (
            f"### {index}. {innovation.name}\n"
            f"Motivation. {innovation.motivation}\n\n"
            f"Design. {innovation.technical_idea}\n\n"
            f"Evidence. {'; '.join(innovation.evidence)}\n\n"
            f"Risk control. {innovation.risk}"
        )

