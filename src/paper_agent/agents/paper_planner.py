"""Paper planner."""

from __future__ import annotations

from paper_agent.state import PaperOutline, PaperRequest, PaperState


class PaperPlannerAgent:
    """Builds a paper narrative around confirmed innovation points."""

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        innovations = state.get("innovations", [])
        experiments = state.get("experiments")
        topic = request.project_name.replace("-", " ").title()
        contribution_names = [item.name for item in innovations]
        contribution_text = "; ".join(contribution_names) or "a targeted method study"
        if experiments and experiments.missing_details:
            central_claim = (
                f"This paper addresses the baseline setting through {contribution_text}, "
                "while reserving empirical improvement claims for verified result tables."
            )
        else:
            central_claim = (
                f"This paper improves the baseline setting through "
                f"{'; '.join(contribution_names) or 'a targeted method improvement'}."
            )
        state["outline"] = PaperOutline(
            title_candidates=[
                f"{topic}: A Venue-Ready Method Improvement Study",
                f"Revisiting {topic} with Innovation-Centered Modeling",
                f"An Evidence-Guided Improvement over the Baseline for {topic}",
            ],
            central_claim=central_claim,
            section_plan={
                "Introduction": [
                    "Establish the task and why the baseline problem matters.",
                    "Identify the baseline gap without overstating it.",
                    "State our innovation points and evidence.",
                ],
                "Related Work": [
                    "Baseline family and direct predecessors.",
                    "Techniques related to each innovation point.",
                    "Positioning against adjacent methods.",
                ],
                "Method": [
                    "Overview of the proposed framework.",
                    "One subsection per innovation point.",
                    "Implementation details only when they clarify the method.",
                ],
                "Experiments": [
                    "Datasets, metrics, baselines.",
                    "Main results table.",
                    "Ablation and sensitivity analysis.",
                    "Qualitative or case-study analysis if available.",
                ],
            },
        )
        return state
