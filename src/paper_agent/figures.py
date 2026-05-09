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


def write_flow_diagram_pdf(
    path: Path,
    *,
    title: str,
    steps: list[str],
    footer: str = "",
) -> Path:
    """Write a compact left-to-right method overview diagram."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = [_short_label(step, 42) for step in steps if str(step).strip()]
    if len(cleaned) < 2:
        raise ValueError("At least two steps are required to generate a flow diagram.")
    cleaned = cleaned[:5]

    box_width = 88
    box_height = 82
    gap = 18
    total_width = len(cleaned) * box_width + (len(cleaned) - 1) * gap
    start_x = (612 - total_width) / 2
    y = 198
    commands: list[str] = [
        "1 1 1 rg 0 0 612 396 re f",
        _text(52, 362, 14, title),
        _text(52, 338, 9, "Evidence-bound pipeline generated from accepted innovation points."),
    ]

    for index, step in enumerate(cleaned):
        x = start_x + index * (box_width + gap)
        commands.extend(
            _rounded_box(
                x,
                y,
                box_width,
                box_height,
                fill="0.91 0.96 0.98",
                stroke="0.14 0.28 0.39",
            )
        )
        commands.extend(_wrapped_text(x + 10, y + 52, 8, step, width=15, max_lines=4))
        commands.append(_text(x + 10, y + 14, 7, f"Stage {index + 1}"))
        if index < len(cleaned) - 1:
            arrow_y = y + box_height / 2
            commands.extend(_arrow(x + box_width + 2, arrow_y, x + box_width + gap - 4, arrow_y))

    if footer:
        commands.extend(_wrapped_text(52, 96, 8, footer, width=78, max_lines=3))

    _write_pdf(path, "\n".join(commands) + "\n")
    return path


def write_prototype_hypergraph_pdf(
    path: Path,
    *,
    title: str,
    notes: list[str] | None = None,
) -> Path:
    """Write a compact prototype-hypergraph schema figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    note_text = " | ".join(_short_label(note, 34) for note in (notes or []) if str(note).strip())
    commands: list[str] = [
        "1 1 1 rg 0 0 612 396 re f",
        _text(52, 362, 14, title),
        _text(52, 338, 9, "Prototype geometry, incidence construction, and hyperedge message passing."),
    ]

    patch_points = [(92, 260), (92, 220), (92, 180), (92, 140)]
    proto_points = [(286, 260), (286, 220), (286, 180), (286, 140)]
    for index, (x, y) in enumerate(patch_points, start=1):
        commands.extend(_circle(x, y, 12, fill="0.96 0.94 0.88", stroke="0.38 0.29 0.12"))
        commands.append(_text(x - 5, y - 3, 7, f"x{index}"))
    for index, (x, y) in enumerate(proto_points, start=1):
        commands.extend(_rounded_box(x - 22, y - 12, 44, 24, fill="0.90 0.95 0.99", stroke="0.13 0.28 0.44"))
        commands.append(_text(x - 9, y - 3, 7, f"p{index}"))

    for patch_x, patch_y in patch_points:
        for proto_x, proto_y in proto_points:
            if abs(patch_y - proto_y) <= 45:
                commands.append(f"0.70 0.74 0.78 RG 0.5 w {patch_x + 13} {patch_y} m {proto_x - 23} {proto_y} l S")

    commands.extend(_rounded_box(36, 74, 112, 42, fill="0.94 0.97 0.94", stroke="0.18 0.38 0.20"))
    commands.extend(_wrapped_text(48, 96, 8, "Patch features and tissue labels", width=18, max_lines=2))
    commands.extend(_arrow(148, 95, 214, 95))
    commands.extend(_rounded_box(214, 74, 130, 42, fill="0.91 0.96 0.98", stroke="0.14 0.28 0.39"))
    commands.extend(_wrapped_text(226, 96, 8, "OT prototype barycenters", width=19, max_lines=2))
    commands.extend(_arrow(344, 95, 410, 95))
    commands.extend(_rounded_box(410, 74, 142, 42, fill="0.97 0.93 0.94", stroke="0.48 0.22 0.26"))
    commands.extend(_wrapped_text(422, 96, 8, "Bidirectional HCoN and risk head", width=20, max_lines=2))

    commands.extend(_rounded_box(380, 186, 150, 74, fill="0.95 0.95 0.98", stroke="0.24 0.24 0.42"))
    commands.extend(_wrapped_text(394, 236, 8, "Hyperedges collect patch-prototype incidence", width=24, max_lines=3))
    commands.extend(_arrow(308, 220, 380, 220))
    commands.extend(_arrow(530, 220, 570, 220))
    commands.append(_text(546, 224, 8, "risk"))

    if note_text:
        commands.extend(_wrapped_text(52, 42, 7, note_text, width=82, max_lines=2))

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


def _wrapped_text(
    x: float,
    y: float,
    size: int,
    text: str,
    *,
    width: int,
    max_lines: int,
) -> list[str]:
    lines = _wrap_words(text, width=width, max_lines=max_lines)
    return [_text(x, y - index * (size + 3), size, line) for index, line in enumerate(lines)]


def _wrap_words(text: str, *, width: int, max_lines: int) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if lines and len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = _short_label(lines[-1], max(4, width))
    return lines or [""]


def _rounded_box(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
) -> list[str]:
    return [
        f"{fill} rg {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f",
        f"{stroke} RG 1 w {x:.2f} {y:.2f} {width:.2f} {height:.2f} re S",
    ]


def _circle(x: float, y: float, radius: float, *, fill: str, stroke: str) -> list[str]:
    k = 0.5522847498 * radius
    path = (
        f"{x + radius:.2f} {y:.2f} m "
        f"{x + radius:.2f} {y + k:.2f} {x + k:.2f} {y + radius:.2f} {x:.2f} {y + radius:.2f} c "
        f"{x - k:.2f} {y + radius:.2f} {x - radius:.2f} {y + k:.2f} {x - radius:.2f} {y:.2f} c "
        f"{x - radius:.2f} {y - k:.2f} {x - k:.2f} {y - radius:.2f} {x:.2f} {y - radius:.2f} c "
        f"{x + k:.2f} {y - radius:.2f} {x + radius:.2f} {y - k:.2f} {x + radius:.2f} {y:.2f} c"
    )
    return [f"{fill} rg {path} f", f"{stroke} RG 1 w {path} S"]


def _arrow(x1: float, y1: float, x2: float, y2: float) -> list[str]:
    return [
        f"0.20 0.24 0.28 RG 1 w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S",
        f"0.20 0.24 0.28 rg {x2:.2f} {y2:.2f} m {x2 - 7:.2f} {y2 + 4:.2f} l {x2 - 7:.2f} {y2 - 4:.2f} l f",
    ]


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
