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
- Optional OpenAI-compatible LLM calls for higher-quality section drafting.
- Draft generation for Abstract, Introduction, Related Work, Method, Experiments framework, and Conclusion.
- LaTeX composition using a built-in template fallback.

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

Configure Volcengine Ark / Doubao, or another OpenAI-compatible provider, in a local `.env`:

```env
ARK_API_KEY=your-ark-api-key
OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/v3
TEXT_MODEL=doubao-seed-1-8-251228
LLM_TIMEOUT_SECONDS=120
LLM_MAX_RETRIES=3
```

`OPENAI_API_KEY` is also supported for OpenAI-compatible providers, but `ARK_API_KEY`
is clearer when using Volcengine Ark. Do not commit `.env`; it is ignored by git.

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

## Design Principles

- Venue handling and LaTeX formatting are first-class responsibilities.
- Method writing is driven by innovation points, not raw code diffs.
- Every contribution should be traceable to a baseline limitation, a method decision, or experiment evidence.
- The reviewer agent should flag overclaiming and missing evidence.
