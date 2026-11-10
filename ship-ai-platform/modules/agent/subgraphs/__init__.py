"""
Multi-Agent Subgraphs Package
각 팀별 세부 서브그래프 모듈 진입점
"""
from .analysis import team1_subgraph
from .search import team2_subgraph
from .docgen import team3_subgraph

# 외부(main_graph.py 등)에서 `from subgraphs import *` 할 때 공개할 목록 정의
__all__ = [
    "team1_subgraph",
    "team2_subgraph",
    "team3_subgraph",
]