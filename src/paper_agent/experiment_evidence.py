"""Experiment-evidence provenance classification."""

from __future__ import annotations


REAL_RESULT_KINDS = {"real_result_file", "provided_result_text", "structured_state"}


def classify_experiment_evidence(
    *,
    source: str = "",
    path: str = "",
    text: str = "",
    result_table_count: int | None = None,
) -> dict[str, object]:
    """Classify whether supplied experiment evidence can support submission claims."""

    source = (source or "").strip()
    path = (path or "").strip()
    text = text or ""
    table_count = int(result_table_count or 0)
    combined = f"{source} {path} {text[:4000]}".lower()
    source_and_path = f"{source} {path}".lower()

    if source == "tcga_cohort_csv" or _contains_any(
        combined,
        [
            "not a model-performance result file",
            "cohort data summary",
            "cohort metadata only",
            "available cohort data only",
        ],
    ):
        return _result(
            "data_only",
            "The input is data/cohort metadata only; add trained-model performance tables before empirical claims.",
        )
    if (
        "mock" in source_and_path
        or "synthetic" in source_and_path
        or _contains_any(
        combined,
        [
            "synthetic mock",
            "mock data",
            "fabricated",
            "pipeline testing only",
            "not a real experiment",
        ],
        )
    ):
        return _result(
            "synthetic_mock",
            "Synthetic or mock experiment numbers are suitable for pipeline testing only; replace them before submission.",
        )
    if source == "inline_demo":
        return _result(
            "demo",
            "Inline demo values exercise the pipeline only; replace them with real experiment results before submission.",
        )
    if not text.strip() and table_count > 0:
        if source == "file":
            return _result(
                "real_result_file",
                "A supplied result file was parsed as trained-model evidence; verify provenance and final numbers.",
            )
        if source == "provided":
            return _result(
                "provided_result_text",
                "Provided experiment text was parsed as trained-model evidence; verify provenance and final numbers.",
            )
        return _result(
            "structured_state",
            "Structured result evidence is present in workflow state; verify its provenance before submission.",
        )
    if not text.strip():
        return _result(
            "missing",
            "No experiment result text was supplied; keep empirical claims out of the draft.",
        )
    if table_count <= 0:
        return _result(
            "unstructured",
            "Experiment text was supplied, but no structured trained-model result table was parsed.",
        )
    if source == "file":
        return _result(
            "real_result_file",
            "A supplied result file was parsed as trained-model evidence; verify provenance and final numbers.",
        )
    return _result(
        "provided_result_text",
        "Provided experiment text was parsed as trained-model evidence; verify provenance and final numbers.",
    )


def is_real_result_evidence(kind: str) -> bool:
    return kind in REAL_RESULT_KINDS


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _result(kind: str, note: str) -> dict[str, object]:
    return {
        "kind": kind,
        "note": note,
        "real_result_evidence": is_real_result_evidence(kind),
    }
