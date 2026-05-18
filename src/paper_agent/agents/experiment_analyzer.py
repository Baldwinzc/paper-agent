"""Experiment result analyzer."""

from __future__ import annotations

import re

from paper_agent.experiment_contract import validate_experiment_contract
from paper_agent.tables import MarkdownTable, extract_markdown_tables
from paper_agent.state import (
    AblationEvidence,
    ExperimentComparison,
    ExperimentSummary,
    ExperimentTableSummary,
    PaperRequest,
    PaperState,
    SensitivityEvidence,
    StatisticalTestEvidence,
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
        ablation_evidence = self._ablation_evidence(raw, metrics)
        sensitivity_evidence = self._sensitivity_evidence(raw, metrics)
        statistical_tests = self._statistical_test_evidence(raw, metrics)
        result_findings = [self._table_result_finding(table) for table in result_tables]
        result_findings = [finding for finding in result_findings if finding]
        observations = self._observations(
            raw,
            result_findings,
            ablation_evidence,
            sensitivity_evidence,
            statistical_tests,
        )
        missing = []
        if not datasets:
            missing.append("Dataset names are not explicit.")
        if not metrics:
            missing.append("Evaluation metrics are not explicit.")
        if not any(table.baseline for table in result_tables) and "baseline" not in raw.lower():
            missing.append("Baseline comparison rows should be made explicit.")

        summary = ExperimentSummary(
            raw_preview=raw[:2000],
            datasets=datasets,
            metrics=metrics,
            result_tables=result_tables,
            ablation_evidence=ablation_evidence,
            sensitivity_evidence=sensitivity_evidence,
            statistical_tests=statistical_tests,
            observations=observations,
            missing_details=missing,
        )
        state["experiments"] = summary
        state.setdefault("artifacts", {})["experiment_result_findings"] = result_findings
        state["artifacts"]["experiment_result_tables"] = [
            table.model_dump() for table in result_tables
        ]
        state["artifacts"]["experiment_ablation_evidence"] = [
            item.model_dump() for item in ablation_evidence
        ]
        state["artifacts"]["experiment_sensitivity_evidence"] = [
            item.model_dump() for item in sensitivity_evidence
        ]
        state["artifacts"]["experiment_statistical_tests"] = [
            item.model_dump() for item in statistical_tests
        ]
        state["artifacts"]["experiment_contract"] = validate_experiment_contract(summary)
        return state

    def _observations(
        self,
        raw: str,
        result_findings: list[str],
        ablation_evidence: list[AblationEvidence],
        sensitivity_evidence: list[SensitivityEvidence],
        statistical_tests: list[StatisticalTestEvidence],
    ) -> list[str]:
        lowered = raw.lower()
        observations = list(result_findings)
        if any(word in lowered for word in ["improve", "gain", "提升", "优于", "better"]):
            observations.append("The provided results suggest an improvement over at least one baseline.")
        if ablation_evidence:
            observations.append(
                f"Ablation evidence includes {len(ablation_evidence)} component comparisons."
            )
        elif any(word in lowered for word in ["ablation", "消融", "w/o", "without"]):
                observations.append("Ablation evidence appears to be available.")
        if sensitivity_evidence:
            observations.extend(
                self._sensitivity_observation(item)
                for item in sensitivity_evidence[:3]
            )
        elif any(word in lowered for word in ["sensitivity", "lambda", "hyperparameter"]):
            observations.append("Sensitivity analysis appears to be available.")
        if statistical_tests:
            significant = sum(1 for item in statistical_tests if item.significant)
            observations.append(
                f"Statistical test evidence includes {len(statistical_tests)} comparisons "
                f"({significant} significant at alpha=0.05)."
            )
        elif re.search(r"\bp\s*[-=<>]\s*0?\.\d+|\bp-value\b|wilcoxon|log-rank", raw, flags=re.I):
            observations.append("Statistical test evidence appears to be available.")
        if any(word in lowered for word in ["case", "visual", "example"]):
            observations.append("Case-study or qualitative evidence appears to be available.")
        return observations or ["Experiment analysis needs more structured result tables."]

    def _sensitivity_observation(self, item: SensitivityEvidence) -> str:
        direction = "maximizes" if item.higher_is_better else "minimizes"
        return (
            f"Sensitivity analysis for {item.parameter} {direction} {item.metric or 'the reported metric'} "
            f"at {item.best_parameter_value} with value {item.best_metric_value:.3f}."
        )

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
            if self._looks_like_statistical_test_table(table):
                continue
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

    def _ablation_evidence(self, raw: str, metrics: list[str]) -> list[AblationEvidence]:
        evidence: list[AblationEvidence] = []
        for table in extract_markdown_tables(raw):
            if not self._looks_like_ablation_table(table):
                continue
            evidence.extend(self._ablation_table_evidence(table, metrics))
        return evidence

    def _looks_like_ablation_table(self, table: MarkdownTable) -> bool:
        source = " ".join([table.caption, *table.headers, *(" ".join(row) for row in table.rows)])
        return bool(
            re.search(
                r"\b(ablation|variant|w/o|without|ablat(?:e|ed|ion)|instead|minus)\b",
                source,
                flags=re.I,
            )
        )

    def _ablation_table_evidence(
        self,
        table: MarkdownTable,
        metrics: list[str],
    ) -> list[AblationEvidence]:
        method_index = self._method_column_index(table.headers)
        reference = self._find_row(
            table.rows,
            method_index,
            ["full", "ours", "proposed", "our method", "paper-agent", "hyper-protosurv"],
        )
        if not reference:
            return []

        reference_name = reference[method_index] if method_index < len(reference) else "Full method"
        default_metric = self._table_metric(table, metrics)
        evidence: list[AblationEvidence] = []
        for column_index, header in enumerate(table.headers):
            if column_index == method_index or re.search(r"\b(delta|diff|change)\b", header, re.I):
                continue
            reference_value = self._numeric_value(
                reference[column_index] if column_index < len(reference) else ""
            )
            if reference_value is None:
                continue
            dataset, metric = self._column_context(header, default_metric)
            higher_is_better = not self._lower_is_better(
                " ".join([header, table.caption]),
                metric,
            )
            for row in table.rows:
                if row is reference:
                    continue
                variant_name = row[method_index] if method_index < len(row) else ""
                if not self._is_ablation_variant(variant_name):
                    continue
                variant_value = self._numeric_value(row[column_index] if column_index < len(row) else "")
                if variant_value is None:
                    continue
                signed_drop = (
                    reference_value - variant_value
                    if higher_is_better
                    else variant_value - reference_value
                )
                evidence.append(
                    AblationEvidence(
                        table_caption=table.caption,
                        dataset=dataset,
                        metric=metric,
                        reference=reference_name,
                        variant=variant_name,
                        reference_value=reference_value,
                        variant_value=variant_value,
                        signed_drop=signed_drop,
                        higher_is_better=higher_is_better,
                        supports=self._ablation_supports(variant_name),
                    )
                )
        return evidence

    def _is_ablation_variant(self, name: str) -> bool:
        return bool(
            re.search(
                r"\b(w/o|without|no|ablat(?:e|ed|ion)|remove(?:d)?|minus|instead)\b",
                name,
                flags=re.I,
            )
        )

    def _ablation_supports(self, variant_name: str) -> list[str]:
        lowered = variant_name.lower()
        supports: list[str] = []
        if re.search(r"\b(ot|wasserstein|adaptive|hyperedge|prototype)\b", lowered):
            supports.append("adaptive hypergraph prototype learning")
        if re.search(r"\b(bidirectional|bhe|hcon|hyperedge update)\b", lowered):
            supports.append("bidirectional hyperedge updates")
        if re.search(r"\b(cross-attention|attention|fusion|mean-pool|pool)\b", lowered):
            supports.append("cross-attention fusion")
        if re.search(r"\b(l_rec|reconstruction|rec|reconstruct)\b", lowered):
            supports.append("reconstruction regularization")
        if re.search(r"\b(loss|objective|cox|survival)\b", lowered):
            supports.append("survival-aware objective")
        return list(dict.fromkeys(supports))

    def _sensitivity_evidence(
        self,
        raw: str,
        metrics: list[str],
    ) -> list[SensitivityEvidence]:
        evidence: list[SensitivityEvidence] = []
        for table in extract_markdown_tables(raw):
            if not self._looks_like_sensitivity_table(table):
                continue
            item = self._sensitivity_table_evidence(table, metrics)
            if item:
                evidence.append(item)
        return evidence

    def _looks_like_sensitivity_table(self, table: MarkdownTable) -> bool:
        source = " ".join([table.caption, *table.headers])
        if re.search(
            r"\b(sensitivity|hyper[-\s]?parameter|parameter|lambda(?:[_-]?[a-z]+)?|λ)\b",
            source,
            flags=re.I,
        ):
            return True
        first_header = table.headers[0] if table.headers else ""
        return bool(
            re.search(
                r"\b(lambda(?:[_-]?[a-z]+)?|λ|alpha|beta|gamma|dropout|weight|k|temperature)\b",
                first_header,
                flags=re.I,
            )
        )

    def _sensitivity_table_evidence(
        self,
        table: MarkdownTable,
        metrics: list[str],
    ) -> SensitivityEvidence | None:
        if len(table.headers) < 2 or len(table.rows) < 2:
            return None

        parameter_index = self._parameter_column_index(table.headers)
        metric_index = self._sensitivity_metric_column_index(table, parameter_index)
        if metric_index is None:
            return None

        values: list[tuple[str, float]] = []
        for row in table.rows:
            if parameter_index >= len(row) or metric_index >= len(row):
                continue
            metric_value = self._numeric_value(row[metric_index])
            if metric_value is None:
                continue
            values.append((row[parameter_index].strip(), metric_value))
        if len(values) < 2:
            return None

        metric_header = table.headers[metric_index]
        dataset, metric = self._column_context(metric_header, self._table_metric(table, metrics))
        higher_is_better = not self._lower_is_better(
            " ".join([metric_header, table.caption]),
            metric,
        )
        best_value, best_metric = (
            max(values, key=lambda item: item[1])
            if higher_is_better
            else min(values, key=lambda item: item[1])
        )
        worst_metric = min(value for _, value in values) if higher_is_better else max(value for _, value in values)
        return SensitivityEvidence(
            table_caption=table.caption,
            parameter=table.headers[parameter_index].strip(),
            metric=metric,
            dataset=dataset,
            best_parameter_value=best_value,
            best_metric_value=best_metric,
            worst_metric_value=worst_metric,
            tested_values=[value for value, _ in values],
            metric_values=[metric_value for _, metric_value in values],
            higher_is_better=higher_is_better,
        )

    def _parameter_column_index(self, headers: list[str]) -> int:
        for index, header in enumerate(headers):
            if re.search(
                r"\b(lambda(?:[_-]?[a-z]+)?|λ|alpha|beta|gamma|parameter|hyper[-\s]?parameter|dropout|weight|k)\b",
                header,
                flags=re.I,
            ):
                return index
        return 0

    def _sensitivity_metric_column_index(
        self,
        table: MarkdownTable,
        parameter_index: int,
    ) -> int | None:
        candidates: list[tuple[int, int]] = []
        for index, header in enumerate(table.headers):
            if index == parameter_index:
                continue
            numeric_rows = sum(
                1
                for row in table.rows
                if index < len(row) and self._numeric_value(row[index]) is not None
            )
            if numeric_rows:
                score = numeric_rows
                if self.METRIC_RE.search(header):
                    score += 10
                candidates.append((score, index))
        if not candidates:
            return None
        return max(candidates)[1]

    def _statistical_test_evidence(
        self,
        raw: str,
        metrics: list[str],
    ) -> list[StatisticalTestEvidence]:
        evidence: list[StatisticalTestEvidence] = []
        for table in extract_markdown_tables(raw):
            evidence.extend(self._statistical_test_table_evidence(table, metrics))
        evidence.extend(self._inline_statistical_tests(raw, metrics))
        return list({self._statistical_test_key(item): item for item in evidence}.values())

    def _statistical_test_table_evidence(
        self,
        table: MarkdownTable,
        metrics: list[str],
    ) -> list[StatisticalTestEvidence]:
        p_index = self._p_value_column_index(table.headers)
        if p_index is None:
            return []
        metric_index = self._metric_column_index(table.headers)
        test_index = self._test_column_index(table.headers)
        comparison_index = self._comparison_column_index(table.headers, p_index, metric_index, test_index)
        evidence: list[StatisticalTestEvidence] = []
        for row in table.rows:
            if p_index >= len(row):
                continue
            parsed = self._parse_p_value(row[p_index])
            if not parsed:
                continue
            p_value_text, p_value = parsed
            metric = ""
            if metric_index is not None and metric_index < len(row):
                metric = self._normalize_metric(row[metric_index])
            if not metric:
                metric = self._table_metric(table, metrics)
            test_name = row[test_index].strip() if test_index is not None and test_index < len(row) else ""
            comparison = (
                row[comparison_index].strip()
                if comparison_index is not None and comparison_index < len(row)
                else self._statistical_comparison_from_row(row, p_index, metric_index, test_index)
            )
            evidence.append(
                StatisticalTestEvidence(
                    table_caption=table.caption,
                    comparison=comparison,
                    metric=metric,
                    test=test_name,
                    p_value=p_value,
                    p_value_text=p_value_text,
                    significant=p_value < 0.05,
                    alpha=0.05,
                )
            )
        return evidence

    def _looks_like_statistical_test_table(self, table: MarkdownTable) -> bool:
        source = " ".join([table.caption, *table.headers])
        return bool(
            self._p_value_column_index(table.headers) is not None
            or re.search(r"\b(statistical|significance|p[-\s]?value|wilcoxon|log-rank)\b", source, flags=re.I)
        )

    def _inline_statistical_tests(
        self,
        raw: str,
        metrics: list[str],
    ) -> list[StatisticalTestEvidence]:
        evidence: list[StatisticalTestEvidence] = []
        for sentence in re.split(r"(?<=[.;])\s+|\n+", raw):
            if not re.search(r"\bp\s*(?:-?value)?\s*[=<>]\s*0?\.\d+", sentence, flags=re.I):
                continue
            parsed = self._parse_p_value(sentence)
            if not parsed:
                continue
            p_value_text, p_value = parsed
            metric_match = self.METRIC_RE.search(sentence)
            test_match = re.search(r"\b(Wilcoxon|log-rank|paired t-test|t-test|bootstrap)\b", sentence, flags=re.I)
            evidence.append(
                StatisticalTestEvidence(
                    comparison=sentence.strip()[:180],
                    metric=self._normalize_metric(metric_match.group(1)) if metric_match else (metrics[0] if metrics else ""),
                    test=test_match.group(1) if test_match else "",
                    p_value=p_value,
                    p_value_text=p_value_text,
                    significant=p_value < 0.05,
                    alpha=0.05,
                )
            )
        return evidence

    def _p_value_column_index(self, headers: list[str]) -> int | None:
        for index, header in enumerate(headers):
            if re.search(r"\bp\s*[-_ ]?value\b|\bp\b", header, flags=re.I):
                return index
        return None

    def _metric_column_index(self, headers: list[str]) -> int | None:
        for index, header in enumerate(headers):
            if re.search(r"\bmetric\b", header, flags=re.I):
                return index
        return None

    def _test_column_index(self, headers: list[str]) -> int | None:
        for index, header in enumerate(headers):
            if re.search(r"\btest\b|\bstatistic\b", header, flags=re.I):
                return index
        return None

    def _comparison_column_index(
        self,
        headers: list[str],
        p_index: int,
        metric_index: int | None,
        test_index: int | None,
    ) -> int | None:
        for index, header in enumerate(headers):
            if index in {p_index, metric_index, test_index}:
                continue
            if re.search(r"\b(comparison|method|model|contrast|pair)\b", header, flags=re.I):
                return index
        for index in range(len(headers)):
            if index not in {p_index, metric_index, test_index}:
                return index
        return None

    def _statistical_comparison_from_row(
        self,
        row: list[str],
        p_index: int,
        metric_index: int | None,
        test_index: int | None,
    ) -> str:
        skipped = {p_index, metric_index, test_index}
        pieces = [value.strip() for index, value in enumerate(row) if index not in skipped and value.strip()]
        return " / ".join(pieces)

    def _parse_p_value(self, text: str) -> tuple[str, float] | None:
        match = re.search(r"(?:p\s*(?:-?value)?\s*)?([<>=])?\s*(0?\.\d+(?:e[-+]?\d+)?)", text, flags=re.I)
        if not match:
            return None
        operator = match.group(1) or "="
        value_text = match.group(2)
        return f"p{operator}{value_text}", float(value_text)

    def _statistical_test_key(self, item: StatisticalTestEvidence) -> tuple[str, str, str]:
        return (item.comparison.lower(), item.metric.upper(), item.p_value_text)

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
        original_header = header.strip()
        metric_match = self.METRIC_RE.search(header)
        metric = self._normalize_metric(metric_match.group(1)) if metric_match else default_metric
        dataset = header
        if metric_match:
            dataset = (header[: metric_match.start()] + header[metric_match.end() :]).strip()
        dataset = re.sub(r"\b(score|value|mean|std|avg|average)\b", " ", dataset, flags=re.I)
        dataset = re.sub(r"[^A-Za-z0-9_-]+", " ", dataset).strip()
        if not dataset and re.search(r"\b(avg|average|mean)\b", original_header, flags=re.I):
            dataset = "Average"
        return dataset or original_header, metric

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
