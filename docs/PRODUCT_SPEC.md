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

## Agent Workflow

```text
InputCollector
  -> BaselineReader
  -> CodeUnderstanding
  -> ExperimentAnalyzer
  -> InnovationAnalyzer
  -> VenueTemplate
  -> PaperPlanner
  -> SectionWriter
  -> LatexComposer
  -> Reviewer
```

## Key Rule

The Method section is not written from "code differences". Code is evidence. Baseline
analysis is evidence. Experiment results are evidence. The Method section is written
from explicit innovation points inferred from those evidence sources and confirmed by
the user when needed.

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

