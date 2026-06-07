"""
agents/heuristic_agent.py
-------------------------
[THÀNH VIÊN 2] Cài đặt HeuristicAgent: di chuyển theo hướng tới goal.

Thuật toán (greedy heuristic, không học):
    1. Giải mã state → (row, col, heading, goal_id)
    2. Lấy vị trí goal tương ứng từ env.GOALS[goal_id]
    3. Ưu tiên FORWARD nếu hướng hiện tại dẫn đến gần goal hơn
    4. Ngược lại rẽ trái/phải về hướng có lợi hơn
    5. STOP nếu đã đứng đúng ô goal

Gợi ý:
    - env.decode_state(state) → (row, col, heading, goal_id)
    - env.GOALS = [(0,6), (3,6), (6,0)]  (goal_id → (row, col))
    - FORWARD=0, TURN_LEFT=1, TURN_RIGHT=2, STOP=3
    - Heading: NORTH=0(↑ row-1), EAST=1(→ col+1), SOUTH=2(↓ row+1), WEST=3(← col-1)

Chạy test:
    python -m pytest tests/test_agents.py::TestHeuristicAgent -v
"""

import numpy as np


class HeuristicAgent:
    """
    Agent đi theo heuristic hướng tới goal.

    Parameters
    ----------
    env : DirectionalCarEnv   Môi trường (cần để decode state và lấy GOALS)
    """

    name = "heuristic"

    def __init__(self, env):
        # TODO: lưu env để dùng env.decode_state() và env.GOALS
        raise NotImplementedError

    def select_action(self, state) -> int:
        """
        Chọn action theo heuristic hướng tới goal.

        Parameters
        ----------
        state : int   State đã mã hóa

        Returns
        -------
        int   Action trong {FORWARD=0, TURN_LEFT=1, TURN_RIGHT=2, STOP=3}
        """
        # TODO:
        #   1. decode state → (row, col, heading, goal_id)
        #   2. lấy (goal_row, goal_col) từ env.GOALS[goal_id]
        #   3. nếu (row, col) == (goal_row, goal_col) → STOP (3)
        #   4. tính hướng mong muốn (desired_heading) để tiến gần goal hơn
        #   5. nếu heading == desired_heading → FORWARD (0)
        #   6. nếu (desired_heading - heading) % 4 == 1 → TURN_RIGHT (2)
        #   7. ngược lại → TURN_LEFT (1)
        raise NotImplementedError

    def update(self, state, action, reward, next_state, terminated, **kwargs):
        """Không học – bỏ qua."""
        pass

    def decay_epsilon(self):
        """Không có epsilon – bỏ qua."""
        pass

    def reset(self):
        """Reset agent về trạng thái ban đầu."""
        pass

    def save(self, path: str):
        """Không có gì để lưu."""
        pass

    def load(self, path: str):
        """Không có gì để tải."""
        pass
