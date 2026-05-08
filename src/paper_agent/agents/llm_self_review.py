"""LLM-backed draft self-review."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from paper_agent.llm import ChatMessage, LLMClient, LLMError
from paper_agent.state import PaperState, ReviewFinding


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
        notes = review.get("section_quality_notes", [])
        artifacts["llm_self_review"] = {
            "mode": "llm",
            "unsupported_claims": claims,
            "section_quality_notes": notes,
            "repaired_from_invalid_json": repaired_from_invalid_json,
        }
        if claims:
            state["review_findings"] = [
                *state.get("review_findings", []),
                *[self._finding_from_claim(item) for item in claims],
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
