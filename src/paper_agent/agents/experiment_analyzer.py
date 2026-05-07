"""Experiment result analyzer."""

from __future__ import annotations

import re

from paper_agent.state import ExperimentSummary, PaperRequest, PaperState


class ExperimentAnalyzerAgent:
    """Extracts a coarse summary from pasted experiment tables or notes."""

    DATASET_RE = re.compile(r"\b([A-Z][A-Za-z0-9_-]*(?:-[A-Za-z0-9_]+)?)\b")
    METRIC_RE = re.compile(
        r"\b(c-index|concordance(?: index)?|ibs|brier(?: score)?|acc(?:uracy)?|f1|auc|mrr|map|bleu|rouge|mae|rmse|psnr|miou)\b",
        re.I,
    )
    DATASET_STOPWORDS = {
        "OT",
        "WSI",
        "WSIS",
        "TCGA",
        "IBS",
        "AUC",
        "C",
        "INDEX",
        "SYNTHETIC",
        "MOCK",
        "EXPERIMENT",
        "RESULTS",
    }

    def run(self, state: PaperState) -> PaperState:
        request: PaperRequest = state["request"]
        raw = request.experiment_results.strip()
        metrics = sorted({self._normalize_metric(m.group(1)) for m in self.METRIC_RE.finditer(raw)})
        datasets = self._extract_datasets(raw, metrics)
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

    def _normalize_metric(self, metric: str) -> str:
        normalized = metric.lower()
        if normalized in {"concordance", "concordance index"}:
            return "C-INDEX"
        if normalized == "brier":
            return "BRIER SCORE"
        return normalized.upper()

    def _extract_datasets(self, raw: str, metrics: list[str]) -> list[str]:
        metric_set = {metric.upper() for metric in metrics}
        candidates = list(dict.fromkeys(self.DATASET_RE.findall(raw)))
        preferred = [
            candidate
            for candidate in candidates
            if candidate.isupper()
            and 2 <= len(candidate) <= 8
            and candidate.upper() not in metric_set
            and candidate.upper() not in self.DATASET_STOPWORDS
        ]
        if preferred:
            return preferred[:8]
        return [
            candidate
            for candidate in candidates
            if candidate.upper() not in metric_set
            and candidate.upper() not in self.DATASET_STOPWORDS
        ][:8]
