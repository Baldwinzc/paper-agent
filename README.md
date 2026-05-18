# paper-agent

Paper-agent is a research writing agent for computer science graduate workflows.

The first milestone focuses on a practical paper-drafting loop:

1. Read a baseline paper PDF.
2. Read our method notes, code summary, target venue, and experiment results.
3. Analyze credible innovation points.
4. Generate a paper plan and draft core sections.
5. Compose the result into the target venue's LaTeX template.

The project is intentionally not a "paper ghostwriter". It is a scientific argument
assistant: code, experiments, and the baseline paper are evidence; the Method section is
written from validated innovation points.

## Current Scope

The current scaffold supports:

- Baseline PDF extraction with a text fallback.
- Lightweight codebase summarization.
- Code-baseline comparison that turns repository-only method evidence into
  innovation seeds without making Method read like a raw diff.
- Experiment-result summarization from pasted Markdown/CSV text.
- Innovation-point analysis.
- Venue template selection and optional remote template download.
- Remote template artifact caching/extraction when a venue exposes a downloadable zip or style file.
- Optional OpenAI-compatible LLM calls for higher-quality section drafting.
- One-shot repair prompts for LLM-written sections rejected by local evidence
  validators.
- Draft generation for Abstract, Introduction, Related Work, Method, Experiments framework, and Conclusion.
- LaTeX composition using a built-in template fallback.
- Overleaf-ready zip export for free Overleaf upload workflows.
- Submission-package validation for `main.tex`, `references.bib`, citation keys,
  referenced graphics, optional local compile checks, and generated zip contents.
- Optional real LaTeX compilation through `latexmk`, `pdflatex`, or lightweight
  `tectonic` when `PAPER_AGENT_RUN_LATEX_COMPILE=1` is set.
- Figure/table planning with labels, captions, target sections, and asset open
  items written to `FIGURE_TABLE_PLAN.md`.
- Dependency-free PDF generation for method overview, prototype-hypergraph,
  main-result, and ablation figures from supplied evidence.
- Markdown experiment table conversion to `booktabs` LaTeX tables.
- Seed BibTeX generation with explicit reviewer warnings for unresolved references.
- Optional OpenAlex/Semantic Scholar metadata resolution with resolved/unresolved
  bibliography verification counts and per-seed resolution traces.
- Technical-query extraction for innovation-derived bibliography seeds, avoiding
  raw `Innovation 1` citation keys.
- Optional OpenAlex related-work discovery from baseline references, papers citing
  the baseline, influential field papers, and recent field papers.
- Related Work citation coverage checks for research-thread subsections.
- Factual consistency checks for unsupported datasets, metrics, experiment numbers,
  and Method subsections not tied to accepted innovation points.
- Optional LLM self-review that reads the evidence bundle and flags unsupported
  factual claims without mutating the draft.
- Draft quality report (`DRAFT_REPORT.md`) and author-facing submission checklist
  (`SUBMISSION_CHECKLIST.md`) included in Overleaf-ready exports.
- Submission-readiness scoring with concrete action items for author review.
- Experiment-evidence provenance classification in run summaries and acceptance
  reports, so synthetic/mock inputs and TCGA cohort metadata are not mistaken
  for real trained-model results.
- Experiment-result contract validation for main result tables, baseline
  comparisons, ablations, sensitivity analysis, and statistical tests.
- JSON run summaries for reproducible smoke runs and showcase artifacts.
- Innovation traceability checks to confirm Method covers accepted contribution points.
- Structured ablation parsing that links variant drops back to likely innovation points.

## Run Locally

```powershell
cd D:\code\agent\paper-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn paper_agent.api:app --reload --port 8000
```

Health check:

```powershell
curl http://localhost:8000/health
```

Configure DeepSeek V4 Pro, or another OpenAI-compatible provider, in a local `.env`:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_API_BASE=https://api.deepseek.com
TEXT_MODEL=deepseek-v4-pro
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=3
LLM_THINKING=disabled
```

`OPENAI_API_BASE` and `OPENAI_API_KEY` are also supported for OpenAI-compatible
providers, and `ARK_API_KEY` is still supported for Volcengine Ark. Do not commit `.env`; it is ignored by git.
`LLM_THINKING=disabled` is the default because DeepSeek V4 thinking mode can return
reasoning content separately from the final `content` field; disabling it keeps the
client compatible with the current section-writing pipeline.
Set `PAPER_AGENT_DISABLE_LLM=1` for deterministic local tests that should not call
the configured model. Set `PAPER_AGENT_DISABLE_TEMPLATE_FETCH=1` to skip remote
template downloads during tests or offline runs.
Set `PAPER_AGENT_DISABLE_LLM_SELF_REVIEW=1` to keep LLM section drafting enabled
while skipping the final LLM reviewer pass.
Set `PAPER_AGENT_DISABLE_REFERENCE_RESOLVE=1` to skip scholarly metadata lookup.
Set `PAPER_AGENT_DISABLE_RELATED_WORK_DISCOVERY=1` to skip related-work expansion
from OpenAlex. Related-work discovery is also skipped when reference resolution is
disabled.
Set `OPENALEX_MAILTO=you@example.com` to identify your OpenAlex API traffic.
`SEMANTIC_SCHOLAR_API_KEY` is optional and only raises Semantic Scholar rate limits.

CLI dry run:

```powershell
paper-agent demo --output outputs/demo
```

If the `paper-agent` command is not available yet, install the project first:

```powershell
cd D:\code\agent\paper-agent
pip install -e .
```

Or run the CLI without installing:

```powershell
$env:PYTHONPATH="D:\code\agent\paper-agent\src"
python -m paper_agent.cli llm-ping
```

Check the configured LLM reviewer on a tiny evidence bundle:

```powershell
$env:PYTHONPATH="D:\code\agent\paper-agent\src"
python -m paper_agent.cli llm-self-review-smoke
```

The smoke command prints the LLM self-review mode and any unsupported claims it
finds. It is useful after changing providers or model names because it exercises
the same JSON review path used at the end of a real draft run.

Run a configured-LLM acceptance smoke on the local Hyper-ProtoSurv materials:

```powershell
$env:PYTHONPATH="D:\code\agent\paper-agent\src"
python -m paper_agent.cli llm-draft-smoke `
  --example-root D:\code\agent\example `
  --experiment-results examples\hyper_protosurv_mock_experiments.md `
  --output-dir outputs\llm-draft-smoke `
  --compile-latex
```

This command uses the configured text model for section drafting and fails if
fewer than four sections are actually written by the LLM. It keeps template and
reference network calls offline by default, so the check isolates whether the
paper-writing path itself can call the model. The command writes both
`RUN_SUMMARY.json` for automation and `ACCEPTANCE_REPORT.md` for a concise
human-readable pass/fail view of the input contract, LLM-written sections,
experiment-source integrity, evidence checks, LaTeX package status, compile
check, and output paths.
The report separates pipeline status from submission evidence status: synthetic
or cohort-metadata inputs can prove the toolchain runs, but they fail the
submission-evidence gate until replaced with real trained-model results.
When `--project-name` is omitted, the smoke command uses the output directory
name as the LaTeX project name, keeping generated artifacts easier to locate.
Add `--strict-results` when the smoke run should fail before any LLM generation
unless the supplied experiment file is real result evidence and satisfies the
experiment-result contract. The same `--no-require-ablation`,
`--no-require-sensitivity`, and `--no-require-statistical-tests` switches are
available when those analyses are intentionally out of scope.
The bundled `examples\hyper_protosurv_mock_experiments.md` file is synthetic
TCGA-style data for this smoke path only; replace it with real experiment tables
for any research draft.

## Acceptance Flow

Every meaningful paper-agent test should exercise the full paper-writing path:

1. Input our code path.
2. Input one baseline paper PDF.
3. Input the target journal or conference.
4. Input experiment results when the draft needs result claims.
5. Output a paper draft, LaTeX project, quality report, run summary, and optional
   Overleaf zip.
6. For configured-LLM smoke runs, output an acceptance report that states whether
   the run passed the project-level contract.

Module-level tests are still useful for debugging, but they do not count as a
paper-generation acceptance test unless this input-output contract is covered.

Draft from local materials:

```powershell
python -m paper_agent.cli draft `
  --project-name hyper-protosurv-tcga `
  --target-venue TPAMI `
  --baseline D:\code\agent\example\baseline `
  --code-path D:\code\agent\example\code\hyper-protosurv `
  --experiment-results D:\code\agent\example\results\tcga_results.md `
  --keyword "whole-slide images" `
  --keyword "survival prediction" `
  --output outputs\hyper-protosurv-tcga\draft.md `
  --zip outputs\hyper-protosurv-tcga-overleaf.zip `
  --summary outputs\hyper-protosurv-tcga\RUN_SUMMARY.json `
  --acceptance-report outputs\hyper-protosurv-tcga\ACCEPTANCE_REPORT.md `
  --online `
  --allow-llm `
  --compile-latex `
  --min-llm-sections 4
```

For the lower-level `draft` command, `--experiment-results` should point to a
real result file supplied by the user. The preferred format is documented in
`docs/EXPERIMENT_RESULTS.md`: use a Markdown table with one method column and
numeric dataset-metric columns such as `BLCA C-index` or `BRCA IBS`. The analyzer
extracts baseline value, proposed-method value, signed improvement, and whether
higher or lower is better. Ablation tables with a `Variant` column are parsed as
full-method versus removed-component evidence and surfaced in the draft report.
Generate a fill-in file with
`python -m paper_agent.cli experiment-template --output examples\tcga_results_template.md`.
Validate a completed result file before full paper generation with
`python -m paper_agent.cli validate-results --experiment-results D:\path\to\tcga_results.md --strict`.
Add `--strict-results` to `paper-agent draft` when you want the same check to
stop paper generation before any LLM or LaTeX work starts.
Add `--require-provenance` when the result file must include a source-artifact
table linking paper numbers to logs, fold-level CSVs, seed records, W&B exports,
or other reproducibility records. Local provenance files are fingerprinted with
SHA-256, and an optional `SHA256` column can enforce checksum matching.
Add `--require-artifact-consistency` when a local CSV provenance artifact should
be used to verify that parsed paper-table values match source rows such as
`method,dataset,metric,value`.
Use `--no-require-ablation`, `--no-require-sensitivity`, or
`--no-require-statistical-tests` when those analyses are outside the paper scope.
For the local Hyper-ProtoSurv TCGA project, the higher-level real-result entry is:

```powershell
paper-agent tcga-draft `
  --example-root D:\code\agent\example `
  --experiment-results D:\code\agent\example\results\tcga_results.md `
  --output-dir outputs\hyper-protosurv-tcga-real `
  --zip outputs\hyper-protosurv-tcga-real-overleaf.zip
```

This command always applies strict real-result preflight before generation and
requires the configured LLM by default. Pass `--disable-llm` only for deterministic
debug runs. The preflight also runs TCGA quality checks for the expected BLCA,
BRCA, LGG, LUAD, and UCEC cohorts, C-index, `Hyper-ProtoSurv`, and `ProtoSurv`;
use the `--expected-*` options to override those defaults.
If you only want to run the built-in local TCGA
showcase, use `sample-hyper-protosurv`; it reads `dataset_csv/*.csv` directly as
cohort metadata, not as performance evidence.

When `draft` writes `--output` or `--summary`, it also writes a Markdown
acceptance report by default: next to the summary when `--summary` is provided,
otherwise next to the draft Markdown. Override the path with
`--acceptance-report`.

Use `--online` or `--offline` to explicitly choose whether template fetching,
reference resolution, and related-work discovery can use network calls. Use
`--allow-llm` or `--disable-llm` to override the local LLM environment for a
single run. `--compile-latex` enables the local LaTeX compiler, and
`--min-llm-sections N` turns a normal draft run into a stricter acceptance run
that fails unless at least `N` sections were actually written by the configured
model.

Add `--skip-llm-self-review` when you want LLM section drafting but do not want
the final second-pass reviewer to call the configured model. The CLI maps this
to the request-level `skip_llm_self_review` flag, so API callers can pass the
same boolean in `/api/papers/draft` JSON payloads. The CLI prints
`LLM self-review: llm`, `unavailable`, `disabled`, or `error` after each draft
run, and the API returns the same summary under `llm_self_review`.

To run the local TCGA showcase with complete synthetic result evidence instead
of cohort metadata only:

```powershell
python -m paper_agent.cli sample-hyper-protosurv `
  --example-root D:\code\agent\example `
  --experiment-results examples\hyper_protosurv_mock_experiments.md `
  --output-dir outputs\hyper-protosurv-tcga-mock `
  --zip outputs\hyper-protosurv-tcga-mock-overleaf.zip `
  --compile-latex
```

This should parse main result tables, ablations, sensitivity analysis, and
statistical tests. It is a pipeline demonstration, not scientific evidence.
If you use the synthetic mock file with `--allow-llm`, keep
`--skip-llm-self-review` for a pipeline demonstration; otherwise the LLM reviewer
may correctly flag mock performance claims as unsupported for real submission.
Add `--strict-results` to make the sample fail before generation when the
experiment input is only TCGA cohort metadata or an incomplete result template.

Use a manually downloaded official template when automatic fetching is blocked:

```powershell
python -m paper_agent.cli draft `
  --project-name hyper-protosurv-tcga `
  --target-venue TPAMI `
  --baseline D:\code\agent\example\baseline `
  --code-path D:\code\agent\example\code\hyper-protosurv `
  --experiment-results D:\code\agent\example\results\tcga_results.md `
  --template-zip D:\path\to\official-template.zip `
  --zip outputs\hyper-protosurv-tcga-overleaf.zip
```

`--template-dir D:\path\to\official-template-folder` is also supported. If the
template contains a sample `main.tex`, paper-agent reuses its preamble and style
assets while replacing the sample body with the generated paper draft.

Showcase the local Hyper-ProtoSurv example in deterministic offline mode:

```powershell
python -m paper_agent.cli sample-hyper-protosurv
```

This reads `D:\code\agent\example`, writes `outputs\hyper-protosurv-sample\draft.md`,
creates `outputs\hyper-protosurv-sample\RUN_SUMMARY.json`, writes
`outputs\hyper-protosurv-sample\ACCEPTANCE_REPORT.md`, and packages an Overleaf
zip. By default, this sample builds its experiment input from the real TCGA
cohort CSV files under
`D:\code\agent\example\code\hyper-protosurv\dataset_csv` and does not fabricate
performance scores. Add `--experiment-results <file>` only when you have real
trained-model result tables to use instead. Add `--allow-llm` to spend
configured model calls, and `--online` to allow remote template/reference
lookups.

For a free Overleaf account, upload the generated zip through
`New Project > Upload Project`. The zip contains `main.tex`, `references.bib`, and
upload notes; add real BibTeX entries before submission.

The bundled `examples/hyper_protosurv_mock_experiments.md` file contains
synthetic TCGA-style mock numbers for parser and end-to-end pipeline testing
only. Do not pass it to the sample command unless you are deliberately testing
mock-result behavior.

## Design Principles

- Venue handling and LaTeX formatting are first-class responsibilities.
- Method writing is driven by innovation points, not raw code diffs.
- Every contribution should be traceable to a baseline limitation, a method decision, or experiment evidence.
- The reviewer agent should flag overclaiming and missing evidence.
