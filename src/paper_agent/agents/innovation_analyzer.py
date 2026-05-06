"""Innovation point analyzer."""

from __future__ import annotations

from paper_agent.state import InnovationPoint, PaperRequest, PaperState


class InnovationAnalyzerAgent:
    """Infers paper-worthy innovation points from evidence sources."""

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        baseline = state.get("baseline")
        code = state.get("code")
        experiments = state.get("experiments")

        notes = request.method_notes.strip()
        innovations: list[InnovationPoint] = []
        if notes:
            chunks = [chunk.strip("- \n") for chunk in notes.split("\n") if chunk.strip()]
            for index, chunk in enumerate(chunks[:3], start=1):
                innovations.append(
                    InnovationPoint(
                        name=f"Innovation {index}: {chunk[:64]}",
                        motivation=(
                            baseline.limitations[0]
                            if baseline and baseline.limitations
                            else "Addresses a limitation identified in the baseline setting."
                        ),
                        technical_idea=chunk,
                        evidence=self._evidence(code, experiments),
                        risk="Needs manual confirmation that the contribution is novel and not overclaimed.",
                    )
                )

        if not innovations:
            innovations.append(
                InnovationPoint(
                    name="Innovation 1: Method improvement to be confirmed",
                    motivation="The baseline leaves room for a targeted method improvement.",
                    technical_idea="User should provide the core technical change in method notes.",
                    evidence=self._evidence(code, experiments),
                    risk="Insufficient method notes; this point should not be treated as final.",
                )
            )

        state["innovations"] = innovations
        return state

    def _evidence(self, code, experiments) -> list[str]:
        evidence = []
        if code and code.summary:
            evidence.append(code.summary)
        if experiments:
            evidence.extend(experiments.observations[:2])
        return evidence or ["Evidence needs to be supplied."]

