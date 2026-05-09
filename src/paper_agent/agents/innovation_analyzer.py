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
        comparison = state.get("artifacts", {}).get("code_baseline_comparison", {})

        notes = request.method_notes.strip()
        innovations: list[InnovationPoint] = []
        claims = []
        if notes:
            claims = [chunk.strip("- \n") for chunk in notes.split("\n") if chunk.strip()]
        elif code and code.method_claims:
            claims = [
                *comparison.get("innovation_seeds", []),
                *code.method_claims,
            ]
        elif comparison.get("innovation_seeds"):
            claims = list(comparison.get("innovation_seeds", []))
        claims = list(dict.fromkeys(claims))

        if claims:
            for index, chunk in enumerate(claims[:4], start=1):
                innovations.append(
                    InnovationPoint(
                        name=f"Innovation {index}: {self._innovation_name(chunk)}",
                        motivation=(
                            baseline.limitations[0]
                            if baseline and baseline.limitations
                            else "Addresses a limitation identified in the baseline setting."
                        ),
                        technical_idea=chunk,
                        evidence=self._evidence(code, experiments, comparison),
                        risk=(
                            "Needs manual confirmation that the contribution is novel and not overclaimed."
                            if notes
                            else "Inferred from repository text; user should confirm novelty and wording."
                        ),
                    )
                )

        if not innovations:
            innovations.append(
                InnovationPoint(
                    name="Innovation 1: Method improvement to be confirmed",
                    motivation="The baseline leaves room for a targeted method improvement.",
                    technical_idea="User should provide the core technical change in method notes.",
                    evidence=self._evidence(code, experiments, comparison),
                    risk="Insufficient method notes; this point should not be treated as final.",
                )
            )

        state["innovations"] = innovations
        return state

    def _evidence(self, code, experiments, comparison=None) -> list[str]:
        evidence = []
        if code and code.summary:
            evidence.append(code.summary)
        if code and code.implementation_evidence:
            evidence.extend(self._select_implementation_evidence(code.implementation_evidence))
        if comparison:
            for shift in comparison.get("likely_method_shifts", [])[:3]:
                technique = shift.get("technique")
                if technique and technique != "problem-setting continuity":
                    evidence.append(f"Innovation support: {technique} is backed by repository evidence.")
                evidence.extend((shift.get("evidence") or [])[:1])
        if code and code.method_claims:
            evidence.extend(code.method_claims[:2])
        if experiments:
            experiment_evidence = [
                observation
                for observation in experiments.observations
                if observation != "Experiment analysis needs more structured result tables."
            ]
            evidence.extend(experiment_evidence[:1])
        return evidence or ["Evidence needs to be supplied."]

    def _select_implementation_evidence(self, implementation_evidence: list[str]) -> list[str]:
        selected: list[str] = []
        labels: set[str] = set()
        for item in implementation_evidence:
            label = self._evidence_label(item)
            if label and label in labels:
                continue
            selected.append(item)
            if label:
                labels.add(label)
            if len(selected) >= 8:
                return selected
        for item in implementation_evidence:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= 8:
                break
        return selected

    def _evidence_label(self, evidence_item: str) -> str:
        if "(" in evidence_item and ")" in evidence_item:
            return evidence_item.split("(", 1)[1].split(")", 1)[0]
        if "[" in evidence_item and "]" in evidence_item:
            return evidence_item.split("[", 1)[1].split("]", 1)[0]
        return ""

    def _innovation_name(self, claim: str) -> str:
        lowered = claim.lower()
        if "adaptive hypergraph prototype" in lowered and "bidirectional hyperedge" in lowered:
            return "Adaptive hypergraph prototype learning with bidirectional updates"
        if "ot-driven adaptive hyperedges" in lowered and "bidirectional hyperedge" in lowered:
            return "OT-driven adaptive hyperedges with bidirectional hyperedge convolution"
        if "online prototype bank" in lowered and "removed" in lowered:
            return "Offline OT prototype construction replacing online prototype regularization"
        if "hcon_beta" in lowered or "loss-weight" in lowered:
            return "Minimal survival-reconstruction training objective"
        if "wasserstein-barycenter" in lowered:
            return "Wasserstein-barycenter prototype geometry"
        if "cross-cluster mask" in lowered:
            return "Cross-cluster OT purity constraint"
        return self._compact_title(claim)

    def _compact_title(self, claim: str, limit: int = 90) -> str:
        clean = claim.split("[", 1)[0].strip()
        for marker in [
            " as reflected by ",
            " as suggested by ",
            " according to ",
            " based on ",
        ]:
            if marker in clean.lower():
                start = clean.lower().find(marker)
                clean = clean[:start].strip()
                break
        clean = clean.rstrip(" ,.;:")
        if len(clean) <= limit:
            return clean
        words = clean.split()
        shortened = ""
        for word in words:
            candidate = f"{shortened} {word}".strip()
            if len(candidate) > limit:
                break
            shortened = candidate
        return (shortened or clean[:limit]).rstrip(" ,.;:")
