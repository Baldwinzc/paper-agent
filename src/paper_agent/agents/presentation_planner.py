"""Figure and table presentation planning."""

from __future__ import annotations

import re
from typing import Any

from paper_agent.state import PaperState
from paper_agent.tables import extract_markdown_tables


class PresentationPlannerAgent:
    """Plans paper figures and tables from available evidence without fabricating assets."""

    def run(self, state: PaperState) -> PaperState:
        state.setdefault("artifacts", {})["presentation_plan"] = self._plan(state)
        return state

    def _plan(self, state: PaperState) -> dict[str, object]:
        figures = self._figure_plan(state)
        tables = self._table_plan(state)
        return {
            "figures": figures,
            "tables": tables,
            "open_items": self._open_items(figures, tables),
        }

    def _figure_plan(self, state: PaperState) -> list[dict[str, object]]:
        figures: list[dict[str, object]] = []
        innovations = state.get("innovations", [])
        experiments = state.get("experiments")
        comparison = state.get("artifacts", {}).get("code_baseline_comparison", {})
        shift_terms = " ".join(comparison.get("code_only_terms", []))
        innovation_text = " ".join(f"{item.name} {item.technical_idea}" for item in innovations)
        evidence_preview = [
            evidence
            for item in innovations
            for evidence in item.evidence[:1]
        ][:3]

        if innovations:
            figures.append(
                self._figure(
                    label="fig:method-overview",
                    title="Method Overview",
                    section="Method",
                    asset_path="figures/method_overview.pdf",
                    caption=(
                        "Overview of the proposed method, organized around the accepted innovation "
                        "points and their evidence-backed data flow."
                    ),
                    evidence=evidence_preview,
                )
            )

        if self._contains_any(
            f"{shift_terms} {innovation_text}",
            ["optimal transport", "wasserstein", "barycenter", "hyperedge", "hypergraph"],
        ):
            figures.append(
                self._figure(
                    label="fig:prototype-hypergraph",
                    title="Adaptive Prototype Hypergraph Construction",
                    section="Method",
                    asset_path="figures/prototype_hypergraph.pdf",
                    caption=(
                        "Adaptive prototype and hypergraph construction used by the proposed method, "
                        "including the evidence-backed geometry and hyperedge update components."
                    ),
                    evidence=self._matching_evidence(innovations, ["transport", "wasserstein", "hypergraph", "hyperedge"]),
                )
            )

        if experiments and experiments.result_tables:
            metrics = ", ".join(experiments.metrics[:3]) or "reported metrics"
            datasets = ", ".join(experiments.datasets[:5]) or "reported datasets"
            figures.append(
                self._figure(
                    label="fig:main-results",
                    title="Main Result Summary",
                    section="Experiments",
                    asset_path="figures/main_results.pdf",
                    caption=(
                        f"Summary visualization of the main results on {datasets} using {metrics}, "
                        "generated only from the supplied experiment tables."
                    ),
                    evidence=[observation for observation in experiments.observations[:3]],
                )
            )

        if experiments and experiments.ablation_evidence:
            figures.append(
                self._figure(
                    label="fig:ablation-summary",
                    title="Ablation Summary",
                    section="Experiments",
                    asset_path="figures/ablation_summary.pdf",
                    caption=(
                        "Component-level ablation summary showing the signed drop for each supplied "
                        "variant relative to the full method."
                    ),
                    evidence=[
                        f"{item.variant}: signed drop {item.signed_drop:+.3f}"
                        for item in experiments.ablation_evidence[:4]
                    ],
                )
            )

        if experiments and experiments.sensitivity_evidence:
            figures.append(
                self._figure(
                    label="fig:sensitivity-summary",
                    title="Sensitivity Summary",
                    section="Experiments",
                    asset_path="figures/sensitivity_summary.pdf",
                    caption=(
                        "Parameter sensitivity summary generated from the supplied sensitivity "
                        "analysis table."
                    ),
                    evidence=[
                        f"{item.parameter}: best {item.best_parameter_value} -> {item.best_metric_value:.3f}"
                        for item in experiments.sensitivity_evidence[:4]
                    ],
                )
            )

        return figures

    def _table_plan(self, state: PaperState) -> list[dict[str, object]]:
        request = state["request"]
        experiments = state.get("experiments")
        markdown_tables = extract_markdown_tables(request.experiment_results or "")
        tables = [
            {
                "label": table.label,
                "caption": table.caption,
                "section": "Experiments",
                "source": "experiment_results",
                "columns": len(table.headers),
                "rows": len(table.rows),
                "status": "rendered_from_markdown",
            }
            for table in markdown_tables
        ]
        if experiments and experiments.ablation_evidence and not any("ablation" in str(item["caption"]).lower() for item in tables):
            tables.append(
                {
                    "label": "tab:ablation-summary",
                    "caption": "Ablation summary from supplied component-level evidence.",
                    "section": "Experiments",
                    "source": "parsed_ablation_evidence",
                    "columns": 4,
                    "rows": len(experiments.ablation_evidence),
                    "status": "planned",
                }
            )
        if experiments and experiments.sensitivity_evidence and not any(
            self._contains_any(str(item["caption"]), ["sensitivity", "parameter", "lambda", "λ"])
            for item in tables
        ):
            tables.append(
                {
                    "label": "tab:sensitivity-summary",
                    "caption": "Sensitivity summary from supplied parameter evidence.",
                    "section": "Experiments",
                    "source": "parsed_sensitivity_evidence",
                    "columns": 4,
                    "rows": len(experiments.sensitivity_evidence),
                    "status": "planned",
                }
            )
        if experiments and experiments.statistical_tests and not any(
            self._contains_any(str(item["caption"]), ["statistical", "p-value", "p value", "significance"])
            for item in tables
        ):
            tables.append(
                {
                    "label": "tab:statistical-tests",
                    "caption": "Statistical-test evidence from supplied p-value rows.",
                    "section": "Experiments",
                    "source": "parsed_statistical_tests",
                    "columns": 5,
                    "rows": len(experiments.statistical_tests),
                    "status": "planned",
                }
            )
        return tables

    def _open_items(self, figures: list[dict[str, object]], tables: list[dict[str, object]]) -> list[str]:
        items = [
            f"Create or attach the planned figure asset `{figure['asset_path']}` for `{figure['label']}`."
            for figure in figures
            if figure.get("status") == "planned"
        ]
        if not tables:
            items.append("Add at least one author-verified result table for the Experiments section.")
        return items[:8]

    def _figure(
        self,
        *,
        label: str,
        title: str,
        section: str,
        asset_path: str,
        caption: str,
        evidence: list[str],
    ) -> dict[str, object]:
        return {
            "label": label,
            "title": title,
            "section": section,
            "asset_path": asset_path,
            "caption": caption,
            "evidence": list(dict.fromkeys(self._clip(item) for item in evidence if item))[:3],
            "status": "planned",
        }

    def _matching_evidence(self, innovations, keywords: list[str]) -> list[str]:
        matches = []
        for innovation in innovations:
            for evidence in innovation.evidence:
                lowered = evidence.lower()
                if any(keyword in lowered for keyword in keywords):
                    matches.append(evidence)
        return matches[:3]

    def _contains_any(self, text: str, needles: list[str]) -> bool:
        lowered = text.lower()
        return any(needle in lowered for needle in needles)

    def _clip(self, text: Any, limit: int = 220) -> str:
        compact = re.sub(r"\s+", " ", str(text)).strip()
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
