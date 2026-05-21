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
When online related-work discovery finds verified candidates, generic keyword
seed references are pruned and the provided baseline seed is enriched from the
matched scholarly record when possible.
The run summary, acceptance report, and top-level research-guide report expose
related-work discovery mode, field query, baseline mention queries,
baseline-lineage/influential/recent candidate counts, and retrieval error
sources plus query-specific error details so literature coverage gaps are explicit.
Their related-work candidate previews should also record a human-readable
discovery-path label and the exact source query used to recover each candidate.
Those same artifacts should expose the planned related-work thread buckets and
the draft's measured thread coverage/alignment status so automation can detect
when a generated Related Work section misses an expected literature bucket.
The standalone `related-work-doctor` command should expose the same discovery
state without requiring a full paper run: field query, baseline-mentioned work
queries, candidate buckets, resolver/discovery error sources, and concrete
rerun commands when online retrieval or better keyword seeds are needed. Its
candidate preview should use the same path/query provenance fields as the full
paper pipeline.
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
falling back to deterministic section text. Section drafting also records a
sanitized call trace with model, token usage, elapsed time, and prompt/output
sizes, but not API keys, prompts, or generated section content.
The generic `paper-e2e-smoke` command fixes the user-level acceptance contract:
baseline PDF, code path, target venue, and experiment-result file must produce a
draft, run summary, acceptance report, optional Overleaf zip, and a
machine-readable `smoke_contract`. Successful runs also write an
`ARTIFACT_MANIFEST.json` with output paths, existence checks, sizes, and hashes
for demo handoff and regression evidence. The `paper-e2e-acceptance` command
runs that full smoke and then writes the compact Markdown showcase report in one
step. The lower-level `paper-e2e-report` command can still turn an existing
manifest plus the run summary and acceptance report into the same report.
For LLM-required runs, successful provider preflight diagnostics are recorded in
the run summary and acceptance report before section drafting evidence.
When its strict result preflight blocks generation, it still writes a blocked
`RUN_SUMMARY.json`, `ACCEPTANCE_REPORT.md`, and `ARTIFACT_MANIFEST.json` so
automation can report contract errors and the next command without parsing
console text. Through `paper-e2e-acceptance`, the same blocked state also gets a
`SHOWCASE_REPORT.md`. The blocked summary includes a `next_actions` repair
chain for validation, artifact-template creation, result generation from
completed CSV artifacts, and smoke rerun.
Completed paper E2E runs should also expose a stable `triage` object in
`RUN_SUMMARY.json` and `ARTIFACT_MANIFEST.json`, and the showcase report should
echo the same `status`, `priority`, and `repair_target` fields for direct
automation-facing ranking.
When an LLM-required smoke blocks during provider preflight, it writes the same
blocked artifact set with sanitized provider diagnostics and repair commands.
With `--write-artifact-template`, the blocked smoke can also write the TODO CSV
artifact templates immediately while still refusing to draft from incomplete
results. With `--generate-results-from-artifacts`, a later smoke run can turn
completed CSV artifacts into strict result Markdown before drafting.
The `tcga-results-guide` command provides the same repair path as a single
entrypoint: it initializes missing result CSV templates, blocks while any CSV
contains TODO placeholders, then runs artifact diagnostics, generates
`tcga_results.md`, and strictly validates the generated result file before the
paper E2E acceptance command is rerun.
The higher-level `research-paper-guide` command links result completion and
paper acceptance: starting from code, baseline PDF, target venue, and either a
strict result file or TCGA result artifacts, it produces the result guide summary,
paper acceptance artifacts, showcase report, and top-level
`RESEARCH_GUIDE_SUMMARY.json` and `RESEARCH_GUIDE_REPORT.md` files.
Those top-level artifacts should include a stable `triage` record with
`status`, `priority`, `priority_rank`, `repair_target`, and `reason`, so
automation can rank `blocked`, `needs_revision`, and `ready` runs directly
from JSON.
The repo should also provide a directory-level `triage-report` command that
scans `RESEARCH_GUIDE_SUMMARY.json` plus standalone `ARTIFACT_MANIFEST.json`
files under one root, deduplicates child manifests already referenced by a
research-guide summary, and writes a ranked JSON/Markdown view for heartbeat or
doctor-style automation. For legacy artifacts that predate stable stored
`triage` fields, the command should derive triage from blocked summary state or
acceptance-triage evidence and record whether each entry was `recorded` or
`derived`.
Its `--results-mode` option makes result handling explicit: `auto` preserves the
default missing/template-result repair flow, `use-existing` requires
`--experiment-results` and never invokes the TCGA result guide, and
`generate-from-artifacts` always reruns artifact-based result generation.
When a child stage blocks, the top-level research-guide artifacts must surface
the inherited `next_actions` repair chain from the blocked result guide or paper
acceptance summary so external automation does not need to inspect nested files.
Those same top-level artifacts should also surface inherited blocking evidence
such as sanitized LLM preflight diagnostics and artifact-template repair status
from the blocked child summary.
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
When checkable CSV artifacts are available, the workflow can compare source rows
against parsed paper result tables by method, dataset, metric, and value before
allowing strict submission-grade validation to pass.
The same consistency check covers main result rows, ablation variant rows, and
sensitivity parameter rows when the CSV exposes matching long-form labels.
Wide CSV result artifacts with method/variant rows and dataset-metric columns are
normalized to the same internal long-form representation before comparison.
Statistical-test artifacts can also be checked by matching comparison, metric,
test name, and p-value.
Fold-level CSV rows are aggregated by method, dataset, and metric so reported
mean values can be checked without requiring users to hand-author an additional
summary CSV.

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
