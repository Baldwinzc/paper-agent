"""Static validation for generated LaTeX submission packages."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from paper_agent.state import PaperState


class SubmissionPackageValidatorAgent:
    """Checks generated LaTeX artifacts before Overleaf upload or submission."""

    STANDARD_CLASSES = {
        "acmart",
        "article",
        "IEEEtran",
        "llncs",
        "neurips",
        "report",
        "standalone",
    }
    GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")

    def run(self, state: PaperState) -> PaperState:
        state.setdefault("artifacts", {})["submission_package"] = self._validate(state)
        return state

    def _validate(self, state: PaperState) -> dict[str, object]:
        errors: list[str] = []
        warnings: list[str] = []
        checks: dict[str, object] = {}

        project_dir = Path(state["latex_project_dir"]) if state.get("latex_project_dir") else None
        main_tex = Path(state["latex_output_path"]) if state.get("latex_output_path") else None
        references = project_dir / "references.bib" if project_dir else None

        if not project_dir or not project_dir.is_dir():
            errors.append("LaTeX project directory is missing.")
            return self._result(errors, warnings, checks)
        checks["project_dir"] = str(project_dir)

        if not main_tex or not main_tex.is_file():
            errors.append("main.tex is missing.")
            return self._result(errors, warnings, checks)
        if references is None or not references.is_file():
            errors.append("references.bib is missing.")
            references_text = ""
        else:
            references_text = references.read_text(encoding="utf-8", errors="ignore")
        tex = main_tex.read_text(encoding="utf-8", errors="ignore")

        checks.update(self._tex_structure_checks(tex, project_dir, errors, warnings))
        checks.update(self._citation_checks(tex, references_text, errors, warnings))
        checks.update(self._graphic_checks(tex, project_dir, errors))
        checks.update(self._bib_quality_checks(references_text, warnings))
        checks["compile"] = self._compile_check(project_dir, main_tex, warnings)
        checks["zip"] = self._zip_check(state, errors, warnings)
        return self._result(errors, warnings, checks)

    def _tex_structure_checks(
        self,
        tex: str,
        project_dir: Path,
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, object]:
        required_patterns = {
            "documentclass": r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}",
            "begin_document": r"\\begin\{document\}",
            "end_document": r"\\end\{document\}",
            "title": r"\\title\{",
            "abstract": r"\\begin\{abstract\}",
            "bibliography": r"\\bibliography\{[^}]+\}|\\addbibresource\{[^}]+\}",
        }
        present = {
            name: bool(re.search(pattern, tex, flags=re.I | re.S))
            for name, pattern in required_patterns.items()
        }
        for name, ok in present.items():
            if not ok:
                errors.append(f"LaTeX structure check failed: missing {name}.")

        class_match = re.search(required_patterns["documentclass"], tex, flags=re.I)
        document_class = class_match.group(1).strip() if class_match else ""
        if document_class and document_class not in self.STANDARD_CLASSES:
            class_file = project_dir / f"{document_class}.cls"
            if not class_file.exists():
                warnings.append(
                    f"Document class `{document_class}` may require a .cls file not found in the package."
                )

        todo_hits = sorted(set(re.findall(r"\b(?:TODO|TBD|PLACEHOLDER)\b", tex, flags=re.I)))
        if todo_hits:
            warnings.append("LaTeX source still contains placeholder markers: " + ", ".join(todo_hits) + ".")

        return {
            "structure": present,
            "document_class": document_class,
            "placeholder_markers": todo_hits,
        }

    def _citation_checks(
        self,
        tex: str,
        references_text: str,
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, object]:
        cite_keys = self._citation_keys(tex)
        bib_keys = self._bib_keys(references_text)
        undefined = [key for key in cite_keys if key not in bib_keys]
        if undefined:
            errors.append("Undefined citation keys in main.tex: " + ", ".join(undefined[:8]) + ".")
        if cite_keys and not bib_keys:
            errors.append("main.tex cites papers but references.bib has no BibTeX entries.")
        if not cite_keys:
            warnings.append("No citation commands were found in main.tex.")
        return {
            "citation_keys": cite_keys,
            "bib_keys": bib_keys,
            "undefined_citation_keys": undefined,
        }

    def _graphic_checks(self, tex: str, project_dir: Path, errors: list[str]) -> dict[str, object]:
        graphics = [
            match.group(1).strip()
            for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
        ]
        missing = [item for item in graphics if not self._graphic_exists(project_dir, item)]
        if missing:
            errors.append("Missing graphics referenced by main.tex: " + ", ".join(missing[:8]) + ".")
        return {"graphics": graphics, "missing_graphics": missing}

    def _bib_quality_checks(self, references_text: str, warnings: list[str]) -> dict[str, object]:
        entries = len(self._bib_keys(references_text))
        todo_years = len(re.findall(r"\byear\s*=\s*\{TODO\}", references_text, flags=re.I))
        placeholder_authors = len(
            re.findall(
                r"\bauthor\s*=\s*\{(?:Baseline authors|Related work authors|To be completed)\}",
                references_text,
                flags=re.I,
            )
        )
        if todo_years:
            warnings.append(f"references.bib contains {todo_years} entries with TODO years.")
        if placeholder_authors:
            warnings.append(f"references.bib contains {placeholder_authors} placeholder author fields.")
        return {
            "bib_entries": entries,
            "bib_todo_years": todo_years,
            "bib_placeholder_authors": placeholder_authors,
        }

    def _compile_check(self, project_dir: Path, main_tex: Path, warnings: list[str]) -> dict[str, object]:
        latexmk = shutil.which("latexmk")
        pdflatex = shutil.which("pdflatex")
        tool = latexmk or pdflatex or ""
        if not tool:
            warnings.append("No local LaTeX compiler was found; static package checks were run only.")
            return {"mode": "not_run", "tool": "", "status": "tool_unavailable"}
        if os.getenv("PAPER_AGENT_RUN_LATEX_COMPILE", "").strip().lower() not in {"1", "true", "yes", "on"}:
            return {"mode": "not_run", "tool": Path(tool).name, "status": "disabled"}

        if latexmk:
            command = [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                main_tex.name,
            ]
        else:
            command = [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                main_tex.name,
            ]
        try:
            completed = subprocess.run(
                command,
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            warnings.append("LaTeX compile check timed out after 60 seconds.")
            return {"mode": "compile", "tool": Path(tool).name, "status": "timeout"}
        status = "passed" if completed.returncode == 0 else "failed"
        if status == "failed":
            warnings.append("LaTeX compile check failed; inspect the generated .log file.")
        return {
            "mode": "compile",
            "tool": Path(tool).name,
            "status": status,
            "returncode": completed.returncode,
        }

    def _zip_check(self, state: PaperState, errors: list[str], warnings: list[str]) -> dict[str, object]:
        zip_path = Path(state["latex_zip_path"]) if state.get("latex_zip_path") else None
        if not zip_path:
            warnings.append("Overleaf zip has not been generated yet.")
            return {"present": False, "path": "", "entries": 0, "contains_main_tex": False}
        if not zip_path.is_file():
            errors.append(f"Overleaf zip path does not exist: {zip_path}.")
            return {"present": False, "path": str(zip_path), "entries": 0, "contains_main_tex": False}
        try:
            with ZipFile(zip_path) as archive:
                names = archive.namelist()
        except BadZipFile:
            errors.append(f"Overleaf zip is not a valid zip file: {zip_path}.")
            return {"present": False, "path": str(zip_path), "entries": 0, "contains_main_tex": False}

        unsafe = [name for name in names if name.startswith("/") or ".." in Path(name).parts]
        if unsafe:
            errors.append("Overleaf zip contains unsafe paths: " + ", ".join(unsafe[:5]) + ".")
        required = {"main.tex", "references.bib"}
        helper_files = {"README_OVERLEAF.md", "TEMPLATE_SOURCE.md"}
        missing = sorted(name for name in required if name not in names)
        missing_helpers = sorted(name for name in helper_files if name not in names)
        if missing:
            errors.append("Overleaf zip is missing required files: " + ", ".join(missing) + ".")
        if missing_helpers:
            warnings.append(
                "Overleaf zip is missing helper notes: " + ", ".join(missing_helpers) + "."
            )
        return {
            "present": True,
            "path": str(zip_path),
            "entries": len(names),
            "contains_main_tex": "main.tex" in names,
            "missing_required_files": missing,
            "missing_helper_files": missing_helpers,
        }

    def _citation_keys(self, tex: str) -> list[str]:
        keys = []
        for match in re.finditer(
            r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}",
            tex,
        ):
            keys.extend(key.strip() for key in match.group(1).split(",") if key.strip())
        return list(dict.fromkeys(keys))

    def _bib_keys(self, references_text: str) -> list[str]:
        return list(
            dict.fromkeys(
                match.group(1).strip()
                for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)", references_text)
            )
        )

    def _graphic_exists(self, project_dir: Path, graphic_path: str) -> bool:
        candidate = project_dir / graphic_path
        if candidate.suffix:
            return candidate.is_file()
        return any(candidate.with_suffix(suffix).is_file() for suffix in self.GRAPHIC_EXTENSIONS)

    def _result(
        self,
        errors: list[str],
        warnings: list[str],
        checks: dict[str, object],
    ) -> dict[str, object]:
        status = "invalid" if errors else "needs_attention" if warnings else "valid"
        return {
            "status": status,
            "errors": list(dict.fromkeys(errors)),
            "warnings": list(dict.fromkeys(warnings)),
            "checks": checks,
        }
