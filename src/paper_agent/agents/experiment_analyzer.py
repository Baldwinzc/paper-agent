"""Experiment result analyzer."""

from __future__ import annotations

import re

from paper_agent.tables import MarkdownTable, extract_markdown_tables
from paper_agent.state import (
    ExperimentComparison,
    ExperimentSummary,
    ExperimentTableSummary,
    PaperRequest,
    PaperState,
)


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
        result_tables = self._table_summaries(raw, metrics)
        result_findings = [self._table_result_finding(table) for table in result_tables]
        result_findings = [finding for finding in result_findings if finding]
        observations = self._observations(raw, result_findings)
        missing = []
        if not datasets:
            missing.append("Dataset names are not explicit.")
        if not metrics:
            missing.append("Evaluation metrics are not explicit.")
        if not any(table.baseline for table in result_tables) and "baseline" not in raw.lower():
            missing.append("Baseline comparison rows should be made explicit.")

        state["experiments"] = ExperimentSummary(
            raw_preview=raw[:2000],
            datasets=datasets,
            metrics=metrics,
            result_tables=result_tables,
            observations=observations,
            missing_details=missing,
        )
        state.setdefault("artifacts", {})["experiment_result_findings"] = result_findings
        state["artifacts"]["experiment_result_tables"] = [
            table.model_dump() for table in result_tables
        ]
        return state

    def _observations(self, raw: str, result_findings: list[str]) -> list[str]:
        lowered = raw.lower()
        observations = list(result_findings)
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

    def _table_summaries(self, raw: str, metrics: list[str]) -> list[ExperimentTableSummary]:
        summaries: list[ExperimentTableSummary] = []
        for table in extract_markdown_tables(raw):
            summary = self._table_summary(table, metrics)
            if summary and summary.comparisons:
                summaries.append(summary)
        return summaries

    def _table_summary(self, table: MarkdownTable, metrics: list[str]) -> ExperimentTableSummary | None:
        method_index = self._method_column_index(table.headers)
        baseline = self._find_row(table.rows, method_index, ["baseline"])
        ours = self._find_row(
            table.rows,
            method_index,
            ["ours", "proposed", "our method", "paper-agent", "hyper-protosurv"],
        )
        if not baseline or not ours:
            return None

        comparisons: list[ExperimentComparison] = []
        method_name = ours[method_index] if method_index < len(ours) else "The proposed method"
        baseline_name = baseline[method_index] if method_index < len(baseline) else "the baseline"
        default_metric = self._table_metric(table, metrics)
        for index, header in enumerate(table.headers):
            if index == method_index:
                continue
            baseline_value = self._numeric_value(baseline[index] if index < len(baseline) else "")
            ours_value = self._numeric_value(ours[index] if index < len(ours) else "")
            if baseline_value is None or ours_value is None:
                continue
            dataset, metric = self._column_context(header, default_metric)
            delta = ours_value - baseline_value
            higher_is_better = not self._lower_is_better(
                " ".join([header, table.caption]),
                metric,
            )
            if not higher_is_better:
                improved = delta < 0
                signed_delta = -delta
            else:
                improved = delta > 0
                signed_delta = delta
            comparisons.append(
                ExperimentComparison(
                    table_caption=table.caption,
                    dataset=dataset,
                    metric=metric,
                    method=method_name,
                    baseline=baseline_name,
                    method_value=ours_value,
                    baseline_value=baseline_value,
                    signed_improvement=signed_delta,
                    higher_is_better=higher_is_better,
                    improved=improved,
                )
            )

        if not comparisons:
            return None

        table_metric = self._dominant_metric(comparisons)
        return ExperimentTableSummary(
            caption=table.caption,
            metric=table_metric,
            method=method_name,
            baseline=baseline_name,
            comparisons=comparisons,
        )

    def _table_metric(self, table: MarkdownTable, metrics: list[str]) -> str:
        source = " ".join([table.caption, *table.headers])
        match = self.METRIC_RE.search(source)
        if match:
            return self._normalize_metric(match.group(1))
        return metrics[0] if metrics else ""

    def _table_result_finding(self, summary: ExperimentTableSummary) -> str:
        comparisons = summary.comparisons
        if not comparisons:
            return ""

        wins = sum(1 for comparison in comparisons if comparison.improved)
        average_delta = sum(comparison.signed_improvement for comparison in comparisons) / len(comparisons)
        direction = "improves over" if wins else "does not improve over"
        return (
            f"{summary.caption}: {summary.method} {direction} {summary.baseline} on "
            f"{wins}/{len(comparisons)} numeric comparisons "
            f"(average signed improvement {average_delta:+.3f})."
        )

    def _column_context(self, header: str, default_metric: str) -> tuple[str, str]:
        metric_match = self.METRIC_RE.search(header)
        metric = self._normalize_metric(metric_match.group(1)) if metric_match else default_metric
        dataset = header
        if metric_match:
            dataset = (header[: metric_match.start()] + header[metric_match.end() :]).strip()
        dataset = re.sub(r"\b(score|value|mean|std|avg|average)\b", " ", dataset, flags=re.I)
        dataset = re.sub(r"[^A-Za-z0-9_-]+", " ", dataset).strip()
        return dataset or header.strip(), metric

    def _dominant_metric(self, comparisons: list[ExperimentComparison]) -> str:
        metrics = [comparison.metric for comparison in comparisons if comparison.metric]
        if not metrics:
            return ""
        return max(dict.fromkeys(metrics), key=metrics.count)

    def _method_column_index(self, headers: list[str]) -> int:
        for index, header in enumerate(headers):
            if re.search(r"\b(method|model|variant|approach)\b", header, flags=re.I):
                return index
        return 0

    def _find_row(self, rows: list[list[str]], method_index: int, keywords: list[str]) -> list[str]:
        for row in rows:
            method = row[method_index].lower() if method_index < len(row) else ""
            if any(keyword in method for keyword in keywords):
                return row
        return []

    def _numeric_value(self, text: str) -> float | None:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        return float(match.group(0))

    def _lower_is_better(self, header: str, metric: str = "") -> bool:
        return bool(
            re.search(
                r"\b(ibs|brier|mae|rmse|loss|error|time|latency)\b",
                f"{header} {metric}",
                flags=re.I,
            )
        )
