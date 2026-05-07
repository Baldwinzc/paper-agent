"""Export helpers for generated paper artifacts."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def zip_latex_project(project_dir: Path, zip_path: Path) -> Path:
    """Create an Overleaf-ready zip from a LaTeX project directory."""

    project_dir = project_dir.resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"LaTeX project directory not found: {project_dir}")

    zip_path = zip_path.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(project_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(project_dir).as_posix())

    return zip_path
