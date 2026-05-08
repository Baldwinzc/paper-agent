"""FastAPI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from paper_agent.state import PaperRequest
from paper_agent.workflow import PaperWorkflow

app = FastAPI(title="paper-agent", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "paper-agent"}


@app.post("/api/papers/draft")
def draft_paper(request: PaperRequest):
    state = PaperWorkflow().run(request)
    return {
        "success": True,
        "outline": state.get("outline"),
        "sections": state.get("sections"),
        "innovations": state.get("innovations"),
        "venue_template": state.get("venue_template"),
        "artifacts": state.get("artifacts", {}),
        "latex_output_path": str(state.get("latex_output_path", "")),
        "review_findings": state.get("review_findings", []),
        "final_markdown": state.get("final_markdown", ""),
    }
