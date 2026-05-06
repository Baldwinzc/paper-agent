"""Experiment result analyzer."""

from __future__ import annotations

import re

from paper_agent.state import ExperimentSummary, PaperRequest, PaperState


class ExperimentAnalyzerAgent:
    """Extracts a coarse summary from pasted experiment tables or notes."""

    DATASET_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]*(?:-[A-Za-z0-9_]+)?)\b")
    METRIC_RE = re.compile(r"\b(acc(?:uracy)?|f1|auc|mrr|map|bleu|rouge|mae|rmse|psnr|miou)\b", re.I)

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        raw = request.experiment_results.strip()
        metrics = sorted({m.group(1).upper() for m in self.METRIC_RE.finditer(raw)})
        datasets = [d for d in dict.fromkeys(self.DATASET_RE.findall(raw)) if d.upper() not in metrics][:8]
        observations = self._observations(raw)
        missing = []
        if not datasets:
            missing.append("Dataset names are not explicit.")
        if not metrics:
            missing.append("Evaluation metrics are not explicit.")
        if "baseline" not in raw.lower():
            missing.append("Baseline comparison rows should be made explicit.")

        state["experiments"] = ExperimentSummary(
            raw_preview=raw[:2000],
            datasets=datasets,
            metrics=metrics,
            observations=observations,
            missing_details=missing,
        )
        return state

    def _observations(self, raw: str) -> list[str]:
        lowered = raw.lower()
        observations = []
        if any(word in lowered for word in ["improve", "gain", "提升", "优于", "better"]):
            observations.append("The provided results suggest an improvement over at least one baseline.")
        if any(word in lowered for word in ["ablation", "消融", "w/o", "without"]):
            observations.append("Ablation evidence appears to be available.")
        if any(word in lowered for word in ["case", "visual", "example"]):
            observations.append("Case-study or qualitative evidence appears to be available.")
        return observations or ["Experiment analysis needs more structured result tables."]

