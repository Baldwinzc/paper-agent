"""Markdown experiment table extraction and LaTeX rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MarkdownTable:
    caption: str
    label: str
    headers: list[str]
    alignments: list[str]
    rows: list[list[str]]


def extract_markdown_tables(markdown: str) -> list[MarkdownTable]:
    """Extract GitHub-flavored Markdown pipe tables with nearby captions."""

    lines = markdown.splitlines()
    tables: list[MarkdownTable] = []
    heading = ""
    recent_text = ""
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            recent_text = ""
            index += 1
            continue
        if line and not line.startswith("|"):
            recent_text = line

        if _looks_like_table_start(lines, index):
            header = _split_pipe_row(lines[index])
            alignments = _parse_alignment_row(lines[index + 1])
            rows: list[list[str]] = []
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = _split_pipe_row(lines[index])
                if len(row) < len(header):
                    row.extend([""] * (len(header) - len(row)))
                rows.append(row[: len(header)])
                index += 1

            caption = _caption_for_table(heading, recent_text, len(tables) + 1)
            tables.append(
                MarkdownTable(
                    caption=caption,
                    label=f"tab:{_slug(caption)}",
                    headers=header,
                    alignments=alignments[: len(header)],
                    rows=rows,
                )
            )
            recent_text = ""
            continue

        index += 1
    return tables


def markdown_table_to_latex(table: MarkdownTable) -> str:
    """Render a parsed table as a booktabs LaTeX table."""

    column_count = len(table.headers)
    environment = "table*" if column_count >= 6 else "table"
    width = r"\textwidth" if environment == "table*" else r"\columnwidth"
    alignment = _alignment_spec(table.alignments, column_count)
    header = " & ".join(_latex_cell(cell) for cell in table.headers) + r" \\"
    body = "\n".join(" & ".join(_latex_cell(cell) for cell in row) + r" \\" for row in table.rows)

    return "\n".join(
        [
            rf"\begin{{{environment}}}[t]",
            r"\centering",
            rf"\caption{{{_latex_text(table.caption)}}}",
            rf"\label{{{table.label}}}",
            rf"\resizebox{{{width}}}{{!}}{{%",
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}%",
            "}",
            rf"\end{{{environment}}}",
        ]
    )


def markdown_tables_to_latex(markdown: str) -> str:
    tables = extract_markdown_tables(markdown)
    return "\n\n".join(markdown_table_to_latex(table) for table in tables)


def _looks_like_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return lines[index].strip().startswith("|") and bool(_parse_alignment_row(lines[index + 1]))


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _parse_alignment_row(line: str) -> list[str]:
    cells = _split_pipe_row(line)
    alignments: list[str] = []
    for cell in cells:
        normalized = cell.replace(" ", "")
        if not re.fullmatch(r":?-{3,}:?", normalized):
            return []
        if normalized.startswith(":") and normalized.endswith(":"):
            alignments.append("c")
        elif normalized.endswith(":"):
            alignments.append("r")
        else:
            alignments.append("l")
    return alignments


def _caption_for_table(heading: str, recent_text: str, index: int) -> str:
    parts = [part.rstrip(".") for part in [heading, recent_text] if part]
    if not parts:
        return f"Experiment results {index}"
    return ". ".join(dict.fromkeys(parts)) + "."


def _alignment_spec(alignments: list[str], column_count: int) -> str:
    if not alignments:
        return "l" + "c" * max(column_count - 1, 0)
    normalized = list(alignments)
    if len(normalized) < column_count:
        normalized.extend(["c"] * (column_count - len(normalized)))
    normalized[0] = "l"
    return "".join(normalized[:column_count])


def _latex_cell(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    placeholder = "<<<PM>>>"
    text = text.replace("+/-", placeholder).replace("±", placeholder)
    escaped = _latex_text(text)
    return escaped.replace(placeholder, r"$\pm$")


def _latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "experiment-results"
