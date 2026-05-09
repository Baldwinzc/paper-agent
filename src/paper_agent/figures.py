"""Small dependency-free PDF chart writer for generated paper figures."""

from __future__ import annotations

import re
from pathlib import Path


def write_bar_chart_pdf(
    path: Path,
    *,
    title: str,
    bars: list[tuple[str, float]],
    y_label: str,
) -> Path:
    """Write a simple single-page PDF bar chart.

    This intentionally avoids heavyweight plotting dependencies. The output is a
    plain vector PDF suitable for `\\includegraphics`.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [(str(label), float(value)) for label, value in bars if value is not None]
    if not cleaned:
        raise ValueError("At least one bar is required to generate a figure.")

    width = 612
    height = 396
    margin_left = 72
    margin_right = 36
    margin_bottom = 72
    margin_top = 54
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    baseline_y = margin_bottom
    max_value = max(abs(value) for _, value in cleaned) or 1.0
    tick_count = 4
    bar_gap = 8
    bar_width = max(12, (chart_width - bar_gap * (len(cleaned) + 1)) / max(1, len(cleaned)))

    commands: list[str] = [
        "1 1 1 rg 0 0 612 396 re f",
        "0.10 0.12 0.16 RG 1 w",
        f"{margin_left} {baseline_y} m {margin_left + chart_width} {baseline_y} l S",
        f"{margin_left} {baseline_y} m {margin_left} {baseline_y + chart_height} l S",
        _text(72, 362, 14, title),
        _text(22, 218, 9, y_label),
    ]

    for tick in range(tick_count + 1):
        value = max_value * tick / tick_count
        y = baseline_y + chart_height * tick / tick_count
        commands.extend(
            [
                "0.82 0.84 0.87 RG 0.5 w",
                f"{margin_left} {y:.2f} m {margin_left + chart_width} {y:.2f} l S",
                "0.10 0.12 0.16 RG 1 w",
                _text(34, y - 3, 8, _format_value(value)),
            ]
        )

    for index, (label, value) in enumerate(cleaned):
        x = margin_left + bar_gap + index * (bar_width + bar_gap)
        bar_height = chart_height * abs(value) / max_value
        y = baseline_y
        commands.extend(
            [
                "0.18 0.38 0.58 rg",
                f"{x:.2f} {y:.2f} {bar_width:.2f} {bar_height:.2f} re f",
                "0.10 0.12 0.16 rg",
                _text(x + 2, y + bar_height + 8, 8, _format_value(value)),
                _rotated_text(x + min(bar_width, 18), 52, 7, _short_label(label)),
            ]
        )

    _write_pdf(path, "\n".join(commands) + "\n")
    return path


def _write_pdf(path: Path, content: str) -> None:
    stream = content.encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 396] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
    xref_offset = sum(len(chunk) for chunk in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    chunks.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        chunks.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    chunks.append(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(b"".join(chunks))


def _text(x: float, y: float, size: int, text: str) -> str:
    return f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({_pdf_text(text)}) Tj ET"


def _rotated_text(x: float, y: float, size: int, text: str) -> str:
    return f"q 0 1 -1 0 {x:.2f} {y:.2f} cm BT /F1 {size} Tf 0 0 Td ({_pdf_text(text)}) Tj ET Q"


def _pdf_text(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _short_label(text: str, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_value(value: float) -> str:
    if abs(value) < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"
