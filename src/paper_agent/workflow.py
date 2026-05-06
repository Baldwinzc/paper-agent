"""Paper-agent workflow."""

from __future__ import annotations

from paper_agent.agents.baseline_reader import BaselineReaderAgent
from paper_agent.agents.code_understanding import CodeUnderstandingAgent
from paper_agent.agents.experiment_analyzer import ExperimentAnalyzerAgent
from paper_agent.agents.innovation_analyzer import InnovationAnalyzerAgent
from paper_agent.agents.latex_composer import LatexComposerAgent
from paper_agent.agents.paper_planner import PaperPlannerAgent
from paper_agent.agents.reviewer import ReviewerAgent
from paper_agent.agents.section_writer import SectionWriterAgent
from paper_agent.agents.venue_template import VenueTemplateAgent
from paper_agent.state import PaperRequest, PaperState


class PaperWorkflow:
    """Sequential MVP workflow, shaped to be migrated to LangGraph."""

    def __init__(self) -> None:
        self.agents = [
            BaselineReaderAgent(),
            CodeUnderstandingAgent(),
            ExperimentAnalyzerAgent(),
            InnovationAnalyzerAgent(),
            VenueTemplateAgent(),
            PaperPlannerAgent(),
            SectionWriterAgent(),
            LatexComposerAgent(),
            ReviewerAgent(),
        ]

    def run(self, request: PaperRequest) -> PaperState:
        state: PaperState = {"request": request, "artifacts": {}}
        for agent in self.agents:
            state = agent.run(state)
        return state

