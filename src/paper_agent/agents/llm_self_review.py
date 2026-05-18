"""LLM-backed draft self-review."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from paper_agent.llm import ChatMessage, LLMClient, LLMError
from paper_agent.state import DraftSections, PaperState, ReviewFinding


class LLMSelfReviewAgent:
    """Uses an LLM as a second-pass reviewer over evidence-grounded draft claims."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def run(self, state: PaperState) -> PaperState:
        artifacts = state.setdefault("artifacts", {})
        if self._disabled(state):
            artifacts["llm_self_review"] = {"mode": "disabled"}
            return state
        if not self.llm_client or not self.llm_client.available:
            artifacts["llm_self_review"] = {"mode": "unavailable"}
            return state
        if not state.get("sections"):
            artifacts["llm_self_review"] = {"mode": "skipped", "reason": "no sections"}
            return state

        repaired_from_invalid_json = False
        try:
            result = self.llm_client.chat(
                self._review_messages(state),
                temperature=0.0,
                max_tokens=2400,
                response_format={"type": "json_object"},
            )
            try:
                review = self._parse_review(result.content)
            except (ValueError, json.JSONDecodeError) as exc:
                repaired_from_invalid_json = True
                repair_result = self.llm_client.chat(
                    self._repair_messages(result.content, str(exc)),
                    temperature=0.0,
                    max_tokens=1800,
                    response_format={"type": "json_object"},
                )
                review = self._parse_review(repair_result.content)
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            artifacts["llm_self_review"] = {"mode": "error", "error": str(exc)}
            return state

        claims = review.get("unsupported_claims", [])
        revisions = self._auto_revise_supported_locations(state, claims)
        rewrite_revisions, rewrite_errors = self._rewrite_unmatched_claims(state, claims)
        revisions = [*revisions, *rewrite_revisions]
        if rewrite_errors:
            for claim in claims:
                claim.setdefault("revision_errors", []).extend(rewrite_errors)
        active_claims = [claim for claim in claims if not claim.get("auto_revised")]
        revised_claims = [claim for claim in claims if claim.get("auto_revised")]
        notes = review.get("section_quality_notes", [])
        artifacts["llm_self_review"] = {
            "mode": "llm",
            "unsupported_claims": active_claims,
            "auto_revised_claims": revised_claims,
            "auto_revisions": revisions,
            "revision_errors": rewrite_errors,
            "section_quality_notes": notes,
            "repaired_from_invalid_json": repaired_from_invalid_json,
        }
        if active_claims:
            state["review_findings"] = [
                *state.get("review_findings", []),
                *[self._finding_from_claim(item) for item in active_claims],
            ]
        return state

    def _disabled(self, state: PaperState) -> bool:
        request = state.get("request")
        if request and request.skip_llm_self_review:
            return True
        value = os.getenv("PAPER_AGENT_DISABLE_LLM_SELF_REVIEW", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _prompt_payload(self, state: PaperState) -> dict[str, Any]:
        request = state["request"]
        baseline = state.get("baseline")
        code = state.get("code")
        experiments = state.get("experiments")
        sections = state["sections"]
        bibliography = state.get("bibliography", [])
        return {
            "task": "Review the draft for unsupported factual claims.",
            "hard_rules": [
                "Check only factual support, not writing style.",
                "A claim is unsupported if it introduces a dataset, metric, number, "
                "baseline, method component, or contribution not present in evidence.",
                "Do not flag cautious statements such as 'requires manual verification'.",
                "Do not flag pending-evaluation placeholders as unsupported claims.",
                "Do not require ablations or metrics to support a method implementation "
                "claim when the code evidence contains that component; require experiments "
                "only for effectiveness, improvement, or performance claims.",
                "Current experiment evidence defines the draft datasets even when they differ "
                "from the baseline paper's datasets; do not flag cohort names solely because "
                "they are not baseline cohorts.",
                "Do not flag cohort counts, slide counts, or planned-evaluation cohort lists when "
                "they appear in the experiment evidence; only flag performance or protocol claims.",
                "Return at most five unsupported_claims.",
                "Keep every JSON string under 160 characters.",
                "Return JSON only.",
            ],
            "output_schema": {
                "unsupported_claims": [
                    {
                        "section": "abstract|introduction|related_work|method|experiments|conclusion",
                        "claim": "short quote or paraphrase",
                        "reason": "why supplied evidence does not support it",
                        "evidence_needed": "what evidence would make it supportable",
                        "severity": "major|minor",
                    }
                ],
                "section_quality_notes": ["brief factual-risk notes, optional"],
            },
            "project_name": request.project_name,
            "target_venue": request.target_venue,
            "evidence": {
                "baseline": self._truncate(baseline.model_dump() if baseline else {}, 3500),
                "code": self._truncate(code.model_dump() if code else {}, 7000),
                "experiments": self._truncate(experiments.model_dump() if experiments else {}, 3500),
                "innovations": self._truncate(
                    [item.model_dump() for item in state.get("innovations", [])],
                    4500,
                ),
                "bibliography_keys": [entry.key for entry in bibliography],
                "related_work_candidates": self._truncate(
                    state.get("artifacts", {}).get("related_work_candidates", []),
                    3000,
                ),
                "rule_based_findings": self._truncate(
                    [finding.model_dump() for finding in state.get("review_findings", [])],
                    2500,
                ),
            },
            "draft_sections": self._truncate(sections.model_dump(), 9000),
        }

    def _review_messages(self, state: PaperState) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=(
                    "You are a strict scientific paper reviewer. You identify only claims "
                    "that are not supported by the supplied evidence. You do not ask for "
                    "stylistic edits. Return compact valid JSON only."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(self._prompt_payload(state), ensure_ascii=False),
            ),
        ]

    def _repair_messages(self, invalid_content: str, error: str) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=(
                    "Repair invalid JSON from a scientific self-review. Return only valid JSON "
                    "matching the requested schema. Keep at most five unsupported_claims and "
                    "short strings."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "parse_error": error,
                        "invalid_json_prefix": invalid_content[:8000],
                        "required_schema": {
                            "unsupported_claims": [
                                {
                                    "section": "section name",
                                    "claim": "short unsupported claim",
                                    "reason": "short reason",
                                    "evidence_needed": "short evidence need",
                                    "severity": "major|minor",
                                }
                            ],
                            "section_quality_notes": ["short note"],
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ]

    def _truncate(self, value: Any, limit: int) -> Any:
        text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        if len(text) <= limit:
            return value
        return text[:limit] + "...[truncated]"

    def _parse_review(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM self-review response must be a JSON object.")
        claims = data.get("unsupported_claims", [])
        notes = data.get("section_quality_notes", [])
        if not isinstance(claims, list):
            raise ValueError("unsupported_claims must be a list.")
        if not isinstance(notes, list):
            raise ValueError("section_quality_notes must be a list.")
        cleaned_claims = [
            self._clean_claim(item)
            for item in claims
            if isinstance(item, dict)
        ]
        data["unsupported_claims"] = [
            claim
            for claim in cleaned_claims
            if claim["claim"] and not self._claim_says_supported(claim)
        ]
        data["section_quality_notes"] = [str(item) for item in notes]
        return data

    def _clean_claim(self, item: dict[str, Any]) -> dict[str, str]:
        severity = str(item.get("severity", "major")).lower()
        if severity not in {"major", "minor"}:
            severity = "major"
        return {
            "section": str(item.get("section", "unknown"))[:80],
            "claim": str(item.get("claim", "")).strip()[:500],
            "reason": str(item.get("reason", "")).strip()[:500],
            "evidence_needed": str(item.get("evidence_needed", "")).strip()[:500],
            "severity": severity,
        }

    def _claim_says_supported(self, claim: dict[str, str]) -> bool:
        reason = claim.get("reason", "").lower()
        evidence_needed = re.sub(r"[^a-z0-9]+", "", claim.get("evidence_needed", "").lower())
        if evidence_needed in {"na", "n/a", "none", "notneeded", "notapplicable"}:
            return True
        if evidence_needed.startswith("noevidenceneeded"):
            return True
        if "not an unsupported factual claim" in reason:
            return True
        if "not an unsupported claim" in reason:
            return True
        if "evidence shows" in reason and "clarif" in claim.get("evidence_needed", "").lower():
            return True
        if self._claim_is_pending_placeholder(claim):
            return True
        if self._claim_only_needs_empirical_validation(claim):
            return True
        supported_phrases = [
            "claim is supported",
            "is supported",
            "well-supported",
            "matching the claim",
            "matches the claim",
            "cautious statement",
            "acceptable",
            "no factual claim",
        ]
        unsupported_phrases = [
            "not supported",
            "unsupported",
            "does not support",
            "lacks support",
            "no evidence",
            "not explicitly stated",
        ]
        if any(phrase in reason for phrase in unsupported_phrases):
            return False
        return any(phrase in reason for phrase in supported_phrases)

    def _claim_is_pending_placeholder(self, claim: dict[str, str]) -> bool:
        text = " ".join([claim.get("claim", ""), claim.get("reason", "")]).lower()
        pending_markers = [
            "remain pending",
            "remains pending",
            "pending evaluation",
            "planned evaluation",
            "planned-evaluation",
            "evaluation is pending",
            "evaluations are pending",
            "is pending",
            "are pending",
            "future experimental",
            "future experiment",
            "will be reported once",
            "placeholder statement",
            "not a factual claim",
        ]
        return any(marker in text for marker in pending_markers)

    def _claim_only_needs_empirical_validation(self, claim: dict[str, str]) -> bool:
        claim_text = claim.get("claim", "").lower()
        evidence_needed = claim.get("evidence_needed", "").lower()
        empirical_requests = [
            "ablation",
            "component analysis",
            "performance",
            "experimental results",
            "experiments comparing",
            "sensitivity analysis",
            "loss curves",
            "metrics",
            "statistical",
        ]
        if not any(marker in evidence_needed for marker in empirical_requests):
            return False
        empirical_claim_markers = [
            "outperform",
            "improve",
            "gain",
            "achieve",
            "obtains",
            "superior",
            "state-of-the-art",
            "competitive",
            "effective",
            "validated",
            "accuracy",
            "c-index",
            "auc",
            "ibs",
        ]
        if any(marker in claim_text for marker in empirical_claim_markers):
            return False
        implementation_markers = [
            "build",
            "construct",
            "consist",
            "combine",
            "objective",
            "loss",
            "module",
            "prototype",
            "hypergraph",
            "hcon",
            "wasserstein",
            "reconstruction",
            "mask",
            "removes",
            "eliminates",
        ]
        return any(marker in claim_text for marker in implementation_markers)

    def _auto_revise_supported_locations(
        self,
        state: PaperState,
        claims: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        sections = state.get("sections")
        if not sections or not claims:
            return []

        values = sections.model_dump()
        revisions: list[dict[str, str]] = []
        for claim in claims:
            section_name = self._section_name(claim.get("section", ""))
            claim_text = claim.get("claim", "").strip()
            if section_name not in values or len(claim_text) < 12:
                continue
            revised_text, removed_text = self._remove_claim_sentence(
                str(values.get(section_name, "")),
                claim_text,
            )
            if not removed_text:
                continue
            values[section_name] = revised_text
            claim["auto_revised"] = True
            claim["revision_action"] = "removed_sentence"
            revisions.append(
                {
                    "section": section_name,
                    "claim": claim_text[:240],
                    "action": "removed_sentence",
                    "removed_text": removed_text[:500],
                }
            )

        if revisions:
            state["sections"] = DraftSections(**values)
        return revisions

    def _rewrite_unmatched_claims(
        self,
        state: PaperState,
        claims: list[dict[str, Any]],
    ) -> tuple[list[dict[str, str]], list[str]]:
        active_claims = [claim for claim in claims if not claim.get("auto_revised")]
        if not active_claims or self._rewrite_disabled():
            return [], []
        try:
            result = self.llm_client.chat(
                self._rewrite_messages(state, active_claims),
                temperature=0.0,
                max_tokens=2400,
                response_format={"type": "json_object"},
            )
            revisions = self._parse_rewrite_result(result.content)
            applied = self._apply_rewrite_revisions(state, active_claims, revisions)
        except Exception as exc:
            return [], [str(exc)]

        return applied, []

    def _rewrite_disabled(self) -> bool:
        value = os.getenv("PAPER_AGENT_DISABLE_LLM_SELF_REWRITE", "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _rewrite_messages(self, state: PaperState, claims: list[dict[str, Any]]) -> list[ChatMessage]:
        sections = state["sections"].model_dump()
        target_sections = sorted({self._section_name(str(claim.get("section", ""))) for claim in claims})
        section_payload = {
            section: sections.get(section, "")
            for section in target_sections
            if section in sections
        }
        return [
            ChatMessage(
                role="system",
                content=(
                    "You conservatively revise scientific paper draft sections. Remove or neutralize "
                    "unsupported factual claims. Do not add datasets, numbers, methods, citations, "
                    "or positive performance claims. Preserve markdown. Return compact valid JSON only."
                ),
            ),
            ChatMessage(
                role="user",
                content=json.dumps(
                    {
                        "task": "Rewrite only sections needed to remove unsupported claims.",
                        "hard_rules": [
                            "Return the full revised text for each changed section.",
                            "Keep supported evidence-grounded content.",
                            "Remove unsupported claims instead of adding new claims.",
                            "If a section cannot be safely rewritten, omit it.",
                        ],
                        "unsupported_claims": claims,
                        "draft_sections": section_payload,
                        "output_schema": {
                            "section_revisions": [
                                {
                                    "section": "section name",
                                    "revised_text": "full revised section text",
                                    "rationale": "short reason for the edit",
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            ),
        ]

    def _parse_rewrite_result(self, content: str) -> list[dict[str, str]]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM self-rewrite response must be a JSON object.")
        raw_revisions = data.get("section_revisions", [])
        if not isinstance(raw_revisions, list):
            raise ValueError("section_revisions must be a list.")
        revisions = []
        for item in raw_revisions:
            if not isinstance(item, dict):
                continue
            revisions.append(
                {
                    "section": str(item.get("section", ""))[:80],
                    "revised_text": str(item.get("revised_text", "")).strip(),
                    "rationale": str(item.get("rationale", "")).strip()[:500],
                }
            )
        return revisions

    def _apply_rewrite_revisions(
        self,
        state: PaperState,
        claims: list[dict[str, Any]],
        revisions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        sections = state.get("sections")
        if not sections or not revisions:
            return []
        values = sections.model_dump()
        claims_by_section: dict[str, list[dict[str, Any]]] = {}
        for claim in claims:
            claims_by_section.setdefault(self._section_name(str(claim.get("section", ""))), []).append(claim)

        applied: list[dict[str, str]] = []
        for revision in revisions:
            section_name = self._section_name(revision.get("section", ""))
            if section_name not in values or section_name not in claims_by_section:
                continue
            original_text = str(values.get(section_name, ""))
            revised_text = revision.get("revised_text", "").strip()
            section_claims = claims_by_section[section_name]
            if not self._rewrite_is_safe(original_text, revised_text, section_claims):
                continue
            values[section_name] = revised_text
            for claim in section_claims:
                claim["auto_revised"] = True
                claim["revision_action"] = "llm_rewrite_section"
            applied.append(
                {
                    "section": section_name,
                    "action": "llm_rewrite_section",
                    "rationale": revision.get("rationale", ""),
                    "original_chars": str(len(original_text)),
                    "revised_chars": str(len(revised_text)),
                }
            )

        if applied:
            state["sections"] = DraftSections(**values)
        return applied

    def _rewrite_is_safe(
        self,
        original_text: str,
        revised_text: str,
        claims: list[dict[str, Any]],
    ) -> bool:
        if not revised_text or self._normalize_for_match(revised_text) == self._normalize_for_match(original_text):
            return False
        if len(revised_text) < max(40, int(len(original_text.strip()) * 0.2)):
            return False
        revised_norm = self._normalize_for_match(revised_text)
        for claim in claims:
            claim_norm = self._normalize_for_match(str(claim.get("claim", "")))
            if claim_norm and claim_norm in revised_norm:
                return False
        return True

    def _section_name(self, section: str) -> str:
        lowered = re.sub(r"[^a-z_]+", "_", section.strip().lower()).strip("_")
        aliases = {
            "related": "related_work",
            "relatedwork": "related_work",
            "related_work": "related_work",
            "intro": "introduction",
            "experiment": "experiments",
            "experimental_results": "experiments",
        }
        return aliases.get(lowered, lowered)

    def _remove_claim_sentence(self, text: str, claim: str) -> tuple[str, str]:
        claim_norm = self._normalize_for_match(claim)
        if not claim_norm:
            return text, ""

        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.lstrip().startswith("#"):
                continue
            line_norm = self._normalize_for_match(line)
            if claim_norm not in line_norm:
                continue
            revised_line, removed = self._remove_sentence_from_line(line, claim, claim_norm)
            if not removed:
                continue
            lines[index] = revised_line
            return self._clean_removed_line_artifacts("\n".join(lines)), removed
        return text, ""

    def _remove_sentence_from_line(self, line: str, claim: str, claim_norm: str) -> tuple[str, str]:
        if self._normalize_for_match(line) == claim_norm:
            return "", line.strip()

        pattern = r"\s+".join(re.escape(part) for part in claim.split())
        direct = re.search(pattern, line, flags=re.I)
        if direct:
            start, end = self._sentence_bounds(line, direct.start(), direct.end())
            revised = line[:start] + line[end:]
            revised = re.sub(r"\s{2,}", " ", revised).strip()
            return revised, line[start:end].strip()

        spans = list(re.finditer(r"[^.!?]+[.!?]?(?:\s+|$)", line))
        if not spans:
            return line, ""
        for match in spans:
            sentence = match.group(0)
            if claim_norm not in self._normalize_for_match(sentence):
                continue
            revised = line[: match.start()] + line[match.end() :]
            revised = re.sub(r"\s{2,}", " ", revised).strip()
            return revised, sentence.strip()
        return line, ""

    def _sentence_bounds(self, line: str, start: int, end: int) -> tuple[int, int]:
        left = 0
        for index in range(start - 1, -1, -1):
            if self._is_sentence_boundary(line, index):
                left = index + 1
                break
        while left < len(line) and line[left].isspace():
            left += 1

        right = end
        if right > 0 and self._is_sentence_boundary(line, right - 1):
            return left, right
        for index in range(end, len(line)):
            if self._is_sentence_boundary(line, index):
                right = index + 1
                break
        return left, right

    def _is_sentence_boundary(self, line: str, index: int) -> bool:
        char = line[index]
        if char not in ".!?":
            return False
        if char == ".":
            previous_is_digit = index > 0 and line[index - 1].isdigit()
            next_is_digit = index + 1 < len(line) and line[index + 1].isdigit()
            if previous_is_digit and next_is_digit:
                return False
        return True

    def _clean_removed_line_artifacts(self, text: str) -> str:
        lines = text.splitlines()
        cleaned: list[str] = []
        blank = False
        for line in lines:
            current_blank = not line.strip()
            if current_blank and blank:
                continue
            cleaned.append(line.rstrip())
            blank = current_blank
        return "\n".join(cleaned).strip()

    def _normalize_for_match(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _finding_from_claim(self, claim: dict[str, str]) -> ReviewFinding:
        section = claim.get("section") or "unknown"
        issue = claim.get("claim") or "Unsupported claim"
        reason = claim.get("reason") or "The LLM self-review could not find support in evidence."
        evidence_needed = claim.get("evidence_needed") or "Add supporting evidence or revise the claim."
        return ReviewFinding(
            severity=claim.get("severity", "major"),  # type: ignore[arg-type]
            issue=f"LLM self-review flagged unsupported claim in {section}: {issue}",
            suggestion=f"{reason} Evidence needed: {evidence_needed}",
        )
