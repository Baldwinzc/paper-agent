# Product Spec

## Product

`paper-agent` helps computer science graduate students turn research materials into a
venue-ready paper scaffold.

## Inputs

- Our code repository or code summary.
- One baseline paper PDF.
- Target conference or journal.
- Experiment results in Markdown, CSV, or pasted text.
- Optional human notes about the method and contribution.

The canonical acceptance flow is: code + baseline paper + target venue +
experiment results -> paper draft. Smaller checks can validate individual agents,
but release-quality testing must cover this full path.

## Outputs

- Title candidates.
- Abstract.
- Introduction.
- Related Work.
- Method.
- Experiments framework.
- Conclusion.
- Reviewer-style critique.
- LaTeX project based on the target venue template.
- JSON run summary for reproducibility and progress inspection.

## Agent Workflow

```text
InputCollector
  -> BaselineReader
  -> CodeUnderstanding
  -> ExperimentAnalyzer
  -> InnovationAnalyzer
  -> VenueTemplate
  -> PaperPlanner
  -> Bibliography
  -> ReferenceResolver
  -> RelatedWorkDiscovery
  -> SectionWriter
  -> EvidenceGuard
  -> LatexComposer
  -> Reviewer
  -> LLMSelfReview
  -> DraftReport
```

## Key Rule

The Method section is not written from "code differences". Code is evidence. Baseline
analysis is evidence. Experiment results are evidence. The Method section is written
from explicit innovation points inferred from those evidence sources and confirmed by
the user when needed.

The Related Work section should not depend only on the user-provided baseline PDF.
The workflow expands it by retrieving the baseline's references, papers that cite the
baseline, influential field papers, and recent field papers before drafting.
When the baseline metadata cannot expose references directly, the workflow parses
numbered references from the baseline PDF and uses named works mentioned in the
baseline's related-work text as OpenAlex search seeds. Candidate papers must match
the mentioned author's surname and overlap with the reference/title context before
they are added to the bibliography.

Bibliography metadata is treated as evidence with status. Automatically resolved
entries are separated from unresolved seed entries, and the draft report keeps both
counts visible so the author can finish reference cleanup before submission.
When bibliography seeds come from innovation points, the system extracts concise
technical search phrases instead of using raw innovation labels as citation queries.
Related Work subsections are checked for real citation coverage so unsupported
research-thread paragraphs are visible in the draft report.
Generated sections are also checked against supplied experiments and innovation
points to flag unsupported datasets, metrics, numeric results, or Method subsections.
When an LLM client is configured, a final LLM self-review pass checks the draft
against the same evidence bundle and records unsupported claims without editing text.
The reviewer pass can be skipped per CLI run, and a dedicated smoke command
exercises the configured provider on a tiny evidence bundle before spending a full
draft-generation call.
The same skip control is represented on `PaperRequest`, so API integrations can
disable only the final reviewer call without changing process-wide environment
variables.
Showcase runs produce a compact JSON summary containing output paths, review
counts, citation/reference status, LaTeX table counts, and LLM self-review mode.
The built-in Hyper-ProtoSurv showcase defaults to TCGA cohort CSV metadata rather
than synthetic performance numbers; trained-model scores must come from a supplied
experiment-results file.

## Phases

### Phase 1

Build the full drafting skeleton with deterministic fallback agents and LaTeX output.

### Phase 2

Add LLM-backed analysis, PDF structure extraction, and stronger citation handling.

### Phase 3

Add code understanding with optional baseline-code comparison, but keep Method writing
centered on innovation points.

### Phase 4

Add rebuttal, revision history, and venue-specific submission checks.
