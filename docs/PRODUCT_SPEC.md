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
  -> CodeBaselineComparison
  -> InnovationAnalyzer
  -> VenueTemplate
  -> PaperPlanner
  -> PresentationPlanner
  -> Bibliography
  -> ReferenceResolver
  -> RelatedWorkDiscovery
  -> SectionWriter
  -> EvidenceGuard
  -> LatexComposer
  -> SubmissionPackageValidator
  -> Reviewer
  -> LLMSelfReview
  -> SubmissionReadiness
  -> DraftReport
```

## Key Rule

The Method section is not written from "code differences". Code is evidence. Baseline
analysis is evidence. Experiment results are evidence. The Method section is written
from explicit innovation points inferred from those evidence sources and confirmed by
the user when needed.

Code-baseline comparison is allowed as an evidence organizer, not as Method prose.
It records shared context, repository-only method candidates, and innovation seeds
so later writing can focus on contribution points instead of describing a raw diff.

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
Each resolver attempt also records a trace containing the seed query, matched
title, source service, DOI, and deduplication target so citation decisions remain
auditable.
When bibliography seeds come from innovation points, the system extracts concise
technical search phrases instead of using raw innovation labels as citation queries.
Related Work subsections are checked for real citation coverage so unsupported
research-thread paragraphs are visible in the draft report.
Generated sections are also checked against supplied experiments and innovation
points to flag unsupported datasets, metrics, numeric results, or Method subsections.
After review, the workflow computes a submission-readiness score across evidence
grounding, writing completeness, citation readiness, and venue packaging, with
explicit blocking items and next actions for the author.
Experiment evidence is classified by provenance. Real result files, structured
workflow-state results, synthetic/mock files, inline demos, and TCGA
cohort-metadata summaries are reported separately so a pipeline smoke run is not
confused with submission-ready empirical evidence.
The experiment-result contract validator reports whether the supplied result file
contains a main method-vs-baseline table plus ablation, sensitivity, and
statistical-test evidence, and the CLI can generate a fill-in Markdown template
for that contract.
The LaTeX submission package is statically validated before reporting: the agent
checks `main.tex`, `references.bib`, citation-key closure, referenced graphics,
required zip entries, and optional local LaTeX compilation when enabled.
Compile validation supports `latexmk`, `pdflatex`, and `tectonic`; `tectonic` is
the preferred lightweight local option because it can fetch TeX assets on demand.
The presentation planner records evidence-bound figure and table plans with
labels, captions, target sections, suggested asset paths, and open items in
`FIGURE_TABLE_PLAN.md` so authors can add visual assets without the draft
inventing figures.
When structured method and experiment evidence is available, the LaTeX composer
can generate simple PDF method overview, prototype-hypergraph, main-result, and
ablation figures and inserts only those figures whose assets exist in the package.
When an LLM client is configured, a final LLM self-review pass checks the draft
against the same evidence bundle and records unsupported claims without editing text.
If an LLM-written section is rejected by local evidence validators, the section
writer makes one targeted repair call with the validation error and the rejected
text before falling back to deterministic prose.
The reviewer pass can be skipped per CLI run, and a dedicated smoke command
exercises the configured provider on a tiny evidence bundle before spending a full
draft-generation call.
The same skip control is represented on `PaperRequest`, so API integrations can
disable only the final reviewer call without changing process-wide environment
variables.
Showcase runs produce a compact JSON summary containing output paths, review
counts, citation/reference status, LaTeX table counts, and LLM self-review mode.
LLM acceptance smoke runs additionally record which paper sections were actually
generated by the configured model, so a full-path test cannot silently pass after
falling back to deterministic section text.
The built-in Hyper-ProtoSurv showcase defaults to TCGA cohort CSV metadata rather
than synthetic performance numbers; trained-model scores must come from a supplied
experiment-results file.
Experiment result files should expose real trained-model tables with method rows
and dataset-metric columns. The analyzer stores per-column comparisons, signed
improvements, and metric direction so generated result prose can be traced back
to exact user-supplied values.
Ablation tables are parsed separately from main results. The workflow stores
variant-vs-full signed drops and maps component names to likely innovation
threads, so Method traceability and draft reports can show which contribution is
supported by which supplied ablation row.
Sensitivity and statistical-test tables are also parsed as structured evidence:
the workflow records best parameter values, tested ranges, exact p-values, test
names, and significance at alpha=0.05. The section writer receives these records
as an evidence contract and must not invent tuning or statistical claims beyond
the supplied rows.
Result-file quality checks can compare parsed datasets, metrics, proposed-method
names, and baseline names against a declared target. The TCGA draft entrypoint
uses BLCA, BRCA, LGG, LUAD, UCEC, C-index, Hyper-ProtoSurv, and ProtoSurv as its
default alignment target before spending LLM calls.
Result-file provenance checks can require a dedicated source-artifact table that
links paper-facing numbers to local logs, fold-level result files, seed records,
or remote tracker exports. Missing local artifacts become blocking evidence
errors when provenance is required. Local artifact files are fingerprinted with
SHA-256 and byte size, and declared checksums are verified when supplied.

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
