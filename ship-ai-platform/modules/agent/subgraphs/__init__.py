"""
Multi-Agent Subgraphs Package
각 팀별 세부 서브그래프 모듈 진입점
"""
from modules.agent.subgraphs.analysis import team1_subgraph
from modules.agent.subgraphs.search import team2_subgraph
from modules.agent.subgraphs.docgen import team3_subgraph
from modules.agent.subgraphs.planning import team4_subgraph

__all__ = [
    "team1_subgraph",
    "team2_subgraph",
    "team3_subgraph",
    "team4_subgraph",
]