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
        if self._disabled():
            artifacts["llm_self_review"] = {"mode": "disabled"}
            return state
        if not self.llm_client or not self.llm_client.available:
            artifacts["llm_self_review"] = {"mode": "unavailable"}
            return state
        if not state.get("sections"):
            artifacts["llm_self_review"] = {"mode": "skipped", "reason": "no sections"}
            return state

        try:
            result = self.llm_client.chat(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You are a strict scientific paper reviewer. You identify only claims "
                            "that are not supported by the supplied evidence. You do not ask for "
                            "stylistic edits."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(self._prompt_payload(state), ensure_ascii=False),
                    ),
                ],
                temperature=0.0,
                max_tokens=1600,
                response_format={"type": "json_object"},
            )
            review = self._parse_review(result.content)
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            artifacts["llm_self_review"] = {"mode": "error", "error": str(exc)}
            return state

        claims = review.get("unsupported_claims", [])
        notes = review.get("section_quality_notes", [])
        artifacts["llm_self_review"] = {
            "mode": "llm",
            "unsupported_claims": claims,
            "section_quality_notes": notes,
        }
        if claims:
            state["review_findings"] = [
                *state.get("review_findings", []),
                *[self._finding_from_claim(item) for item in claims],
            ]
        return state

    def _disabled(self) -> bool:
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
                "code": self._truncate(code.model_dump() if code else {}, 3500),
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
        data["unsupported_claims"] = [
            self._clean_claim(item)
            for item in claims
            if isinstance(item, dict)
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
