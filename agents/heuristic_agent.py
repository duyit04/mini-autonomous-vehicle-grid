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
        self.env = env

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
        row, col, heading, goal_id = self.env.state_decoder(state)
        goal_row, goal_col = self.env.GOALS[goal_id]

        if (row, col) == (goal_row, goal_col):
            return 3  # STOP

        dr = goal_row - row
        dc = goal_col - col

        # Xây danh sách hướng ưu tiên: trục xa hơn trước, trục gần hơn sau
        if abs(dr) >= abs(dc):
            primary   = 2 if dr > 0 else 0   # SOUTH/NORTH
            secondary = 1 if dc > 0 else 3   # EAST/WEST  (dc==0 thì không dùng)
        else:
            primary   = 1 if dc > 0 else 3   # EAST/WEST
            secondary = 2 if dr > 0 else 0   # SOUTH/NORTH (dr==0 thì không dùng)

        # Lọc: chỉ giữ hướng mà ô phía trước không bị chặn
        candidates = [primary]
        if dc != 0 and secondary != primary:
            candidates.append(secondary)
        # Thêm 2 hướng còn lại làm fallback tránh kẹt hoàn toàn
        for h in range(4):
            if h not in candidates:
                candidates.append(h)

        def _blocked(h):
            """True nếu tiến theo hướng h sẽ va chạm tường hoặc obstacle."""
            ddr, ddc = self.env.DELTAS[h]
            nr, nc = row + ddr, col + ddc
            G = self.env.GRID_SIZE
            return not (0 <= nr < G and 0 <= nc < G) or (nr, nc) in self.env.OBSTACLES

        # Chọn hướng tốt nhất không bị chặn
        desired_heading = None
        for h in candidates:
            if not _blocked(h):
                desired_heading = h
                break

        # Tất cả hướng đều bị chặn (không thể xảy ra trong map liên thông)
        if desired_heading is None:
            return 1  # TURN_LEFT để thoát kẹt

        if heading == desired_heading:
            return 0  # FORWARD

        if (desired_heading - heading) % 4 == 1:
            return 2  # TURN_RIGHT

        return 1  # TURN_LEFT

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
