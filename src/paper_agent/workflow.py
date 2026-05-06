"""Paper-agent workflow."""

from __future__ import annotations

from paper_agent.agents.baseline_reader import BaselineReaderAgent
from paper_agent.agents.code_understanding import CodeUnderstandingAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.agents.evidence_guard import EvidenceGuardAgent
from paper_agent.agents.innovation_analyzer import InnovationAnalyzerAgent
from paper_agent.agents.latex_composer import LatexComposerAgent
from paper_agent.agents.paper_planner import PaperPlannerAgent
from paper_agent.agents.reviewer import ReviewerAgent
from paper_agent.agents.section_writer import SectionWriterAgent
from paper_agent.agents.venue_template import VenueTemplateAgent
from paper_agent.config import load_llm_config
from paper_agent.llm import LLMClient
from paper_agent.state import PaperRequest, PaperState


class PaperWorkflow:
    """Sequential MVP workflow, shaped to be migrated to LangGraph."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        if llm_client is None:
            config = load_llm_config()
            llm_client = LLMClient(config) if config.configured else None
        self.llm_client = llm_client
        self.agents = [
            BaselineReaderAgent(),
            CodeUnderstandingAgent(),
            ExperimentAnalyzerAgent(),
            InnovationAnalyzerAgent(),
            VenueTemplateAgent(),
            PaperPlannerAgent(),
            SectionWriterAgent(llm_client=llm_client),
            EvidenceGuardAgent(),
            LatexComposerAgent(),
            ReviewerAgent(),
        ]

    def run(self, request: PaperRequest) -> PaperState:
        state: PaperState = {"request": request, "artifacts": {}}
        for agent in self.agents:
            state = agent.run(state)
        return state
