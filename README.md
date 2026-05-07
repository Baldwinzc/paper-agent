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
- Experiment-result summarization from pasted Markdown/CSV text.
- Innovation-point analysis.
- Venue template selection and optional remote template download.
- Remote template artifact caching/extraction when a venue exposes a downloadable zip or style file.
- Optional OpenAI-compatible LLM calls for higher-quality section drafting.
- Draft generation for Abstract, Introduction, Related Work, Method, Experiments framework, and Conclusion.
- LaTeX composition using a built-in template fallback.
- Overleaf-ready zip export for free Overleaf upload workflows.
- Markdown experiment table conversion to `booktabs` LaTeX tables.
- Seed BibTeX generation with explicit reviewer warnings for unverified references.

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
```

`OPENAI_API_BASE` and `OPENAI_API_KEY` are also supported for OpenAI-compatible
providers, and `ARK_API_KEY` is still supported for Volcengine Ark. Do not commit `.env`; it is ignored by git.
Set `PAPER_AGENT_DISABLE_LLM=1` for deterministic local tests that should not call
the configured model. Set `PAPER_AGENT_DISABLE_TEMPLATE_FETCH=1` to skip remote
template downloads during tests or offline runs.

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

Draft from local materials:

```powershell
python -m paper_agent.cli draft `
  --project-name hyper-protosurv-mock `
  --target-venue TPAMI `
  --baseline D:\code\agent\example\baseline `
  --code-path D:\code\agent\example\code\hyper-protosurv `
  --experiment-results examples\hyper_protosurv_mock_experiments.md `
  --keyword "whole-slide images" `
  --keyword "survival prediction" `
  --output outputs\hyper-protosurv-mock\draft.md `
  --zip outputs\hyper-protosurv-mock-overleaf.zip
```

Use a manually downloaded official template when automatic fetching is blocked:

```powershell
python -m paper_agent.cli draft `
  --project-name hyper-protosurv-mock `
  --target-venue TPAMI `
  --baseline D:\code\agent\example\baseline `
  --code-path D:\code\agent\example\code\hyper-protosurv `
  --experiment-results examples\hyper_protosurv_mock_experiments.md `
  --template-zip D:\path\to\official-template.zip `
  --zip outputs\hyper-protosurv-mock-overleaf.zip
```

`--template-dir D:\path\to\official-template-folder` is also supported. If the
template contains a sample `main.tex`, paper-agent reuses its preamble and style
assets while replacing the sample body with the generated paper draft.

For a free Overleaf account, upload the generated zip through
`New Project > Upload Project`. The zip contains `main.tex`, `references.bib`, and
upload notes; add real BibTeX entries before submission.

The bundled `examples/hyper_protosurv_mock_experiments.md` file contains synthetic
mock numbers for pipeline testing only. Replace it with real experiment tables before
using generated text in a paper.

## Design Principles

- Venue handling and LaTeX formatting are first-class responsibilities.
- Method writing is driven by innovation points, not raw code diffs.
- Every contribution should be traceable to a baseline limitation, a method decision, or experiment evidence.
- The reviewer agent should flag overclaiming and missing evidence.
