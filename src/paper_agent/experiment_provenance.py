"""Experiment-result provenance checks."""

from __future__ import annotations

import re
from pathlib import Path

from paper_agent.tables import MarkdownTable, extract_markdown_tables


REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://", "oss://", "wandb://")


def assess_experiment_provenance(
    raw: str,
    *,
    result_path: Path | None = None,
    require_provenance: bool = False,
) -> dict[str, object]:
    """Check whether a result file declares source artifacts for its numbers."""

    tables = [
        table
        for table in extract_markdown_tables(raw)
        if _looks_like_provenance_table(table)
    ]
    if not tables:
        status = "invalid" if require_provenance else "needs_attention"
        return {
            "status": status,
            "errors": ["Missing result provenance table."] if require_provenance else [],
            "warnings": [
                "No result provenance table was found; add logs, fold files, seeds, or export artifacts."
            ],
            "entries": [],
            "checks": {
                "tables": 0,
                "entries": 0,
                "local_paths": 0,
                "remote_references": 0,
                "missing_paths": 0,
            },
        }

    entries = []
    errors: list[str] = []
    warnings: list[str] = []
    for table in tables:
        path_index = _path_column_index(table)
        if path_index is None:
            errors.append(f"Provenance table `{table.caption}` must contain a path/file/source/url column.")
            continue
        name_index = _name_column_index(table, path_index)
        for row in table.rows:
            if path_index >= len(row):
                continue
            raw_path = _clean_cell(row[path_index])
            if not raw_path:
                continue
            entry = _entry(
                raw_path,
                result_path=result_path,
                table=table,
                name=_clean_cell(row[name_index]) if name_index is not None and name_index < len(row) else "",
                row_text=" ".join(row),
            )
            entries.append(entry)
            if entry["kind"] == "local" and not entry["exists"]:
                errors.append(f"Missing provenance artifact: {entry['path']}.")

    if not entries:
        errors.append("Provenance tables were found, but no artifact paths were parsed.")
    if entries and not any(entry.get("seed") or entry.get("fold") for entry in entries):
        warnings.append("No seed or fold identifiers were parsed from provenance entries.")

    local_paths = [entry for entry in entries if entry["kind"] == "local"]
    remote_refs = [entry for entry in entries if entry["kind"] == "remote"]
    missing_paths = [entry for entry in local_paths if not entry["exists"]]
    return {
        "status": "invalid" if errors else "needs_attention" if warnings else "complete",
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "entries": entries,
        "checks": {
            "tables": len(tables),
            "entries": len(entries),
            "local_paths": len(local_paths),
            "remote_references": len(remote_refs),
            "missing_paths": len(missing_paths),
        },
    }


def _looks_like_provenance_table(table: MarkdownTable) -> bool:
    source = " ".join([table.caption, *table.headers]).lower()
    return any(
        term in source
        for term in [
            "provenance",
            "source artifact",
            "source artifacts",
            "result source",
            "result sources",
            "training log",
            "reproducibility",
        ]
    )


def _path_column_index(table: MarkdownTable) -> int | None:
    preferred = ["path", "file", "log", "source", "uri", "url"]
    headers = [header.lower() for header in table.headers]
    for term in preferred:
        for index, header in enumerate(headers):
            if term in header:
                return index
    for index, header in enumerate(headers):
        if "artifact" in header:
            return index
    return None


def _name_column_index(table: MarkdownTable, path_index: int) -> int | None:
    for index, header in enumerate(table.headers):
        lowered = header.lower()
        if index != path_index and any(term in lowered for term in ["artifact", "name", "run"]):
            return index
    return 0 if path_index != 0 and table.headers else None


def _entry(
    raw_path: str,
    *,
    result_path: Path | None,
    table: MarkdownTable,
    name: str,
    row_text: str,
) -> dict[str, object]:
    seed = _extract_token(row_text, "seed")
    fold = _extract_token(row_text, "fold")
    if raw_path.lower().startswith(REMOTE_PREFIXES):
        return {
            "name": name,
            "path": raw_path,
            "kind": "remote",
            "exists": True,
            "resolved_path": "",
            "table": table.caption,
            "seed": seed,
            "fold": fold,
        }

    resolved = _resolve_local_path(raw_path, result_path)
    return {
        "name": name,
        "path": raw_path,
        "kind": "local",
        "exists": bool(resolved and resolved.exists()),
        "resolved_path": str(resolved) if resolved else "",
        "table": table.caption,
        "seed": seed,
        "fold": fold,
    }


def _resolve_local_path(raw_path: str, result_path: Path | None) -> Path | None:
    path = Path(raw_path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if result_path:
            candidates.append(result_path.parent / path)
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _clean_cell(text: str) -> str:
    cleaned = text.strip().strip("`")
    match = re.search(r"\[([^\]]+)\]\(([^)]+)\)", cleaned)
    if match:
        return match.group(2).strip()
    return cleaned


def _extract_token(text: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*[:=]\s*([A-Za-z0-9_.-]+)", text, flags=re.I)
    return match.group(1) if match else ""
