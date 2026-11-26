# modules/agent/run_test.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.agent.subgraphs import team1_subgraph, team2_subgraph, team3_subgraph, team4_subgraph
from modules.agent.tools.cad_tools import confirm_cad_work_plan

# 1) Team 4 export 확인
assert team4_subgraph is not None

# 2) 잘못된 confirm_mode 즉시 거부 확인
try:
    confirm_cad_work_plan(drawing_meta={}, edited_packages=[], confirm_mode="INVALID_MODE", user_engine=object())
    assert False, "예외가 발생해야 합니다"
except ValueError as e:
    print("confirm_mode 검증 통과:", e)