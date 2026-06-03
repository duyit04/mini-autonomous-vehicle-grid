"""
heuristic_agent.py
------------------
Agent heuristic: luôn quay về hướng làm giảm Manhattan distance đến goal,
sau đó tiến thẳng. Có tránh obstacle cơ bản (nếu phía trước bị chặn,
quay phải để thoát).

Không có cơ chế học – dùng làm baseline trung gian.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.custom_env import DirectionalCarEnv


class HeuristicAgent:
    """
    Heuristic goal-directed agent.

    Chiến lược:
        1. Nếu xe đang ở goal → STOP.
        2. Tính desired_heading để giảm Manhattan distance.
        3. Nếu heading hiện tại != desired_heading → quay (TURN_LEFT hoặc TURN_RIGHT).
        4. Nếu heading == desired_heading:
            - Nếu ô phía trước trống → FORWARD.
            - Nếu ô phía trước bị chặn (tường/obstacle) → TURN_RIGHT (thử thoát).

    Parameters
    ----------
    env : DirectionalCarEnv
        Cần truy cập GOALS, OBSTACLES, DELTAS, GRID_SIZE và state_decoder().
    """

    def __init__(self, env: "DirectionalCarEnv"):
        self.env  = env
        self.name = "Heuristic"

    # ------------------------------------------------------------------ #

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Chọn hành động dựa trên heuristic.

        Parameters
        ----------
        state : int
            Trạng thái đã mã hóa.
        eval_mode : bool
            Bỏ qua – heuristic luôn deterministic.

        Returns
        -------
        int
            Hành động: FORWARD / TURN_LEFT / TURN_RIGHT / STOP.
        """
        r, c, heading, goal_id = self.env.state_decoder(state)
        goal_r, goal_c = self.env.GOALS[goal_id]

        # --- Đã đến goal → STOP ---
        if (r, c) == (goal_r, goal_c):
            return self.env.STOP

        dr = goal_r - r
        dc = goal_c - c

        # --- Tính desired_heading ---
        if abs(dr) >= abs(dc):
            desired = self.env.SOUTH if dr > 0 else self.env.NORTH
        else:
            desired = self.env.EAST if dc > 0 else self.env.WEST

        # --- Tính góc quay cần thiết ---
        diff = (desired - heading) % 4

        if diff == 0:
            # Đang quay đúng hướng – kiểm tra phía trước
            fwd_r = r + self.env.DELTAS[heading][0]
            fwd_c = c + self.env.DELTAS[heading][1]

            path_clear = (
                0 <= fwd_r < self.env.GRID_SIZE
                and 0 <= fwd_c < self.env.GRID_SIZE
                and (fwd_r, fwd_c) not in self.env.OBSTACLES
            )
            if path_clear:
                return self.env.FORWARD
            else:
                # Bị chặn → thử quay phải để thoát
                return self.env.TURN_RIGHT

        elif diff == 1:
            # Cần quay phải 1 lần
            return self.env.TURN_RIGHT

        elif diff == 3:
            # Cần quay trái 1 lần (ngắn hơn quay phải 3 lần)
            return self.env.TURN_LEFT

        else:
            # diff == 2: ngược chiều hoàn toàn → quay phải
            return self.env.TURN_RIGHT

    # ------------------------------------------------------------------ #

    def update(self, *args, **kwargs):
        """Không học – bỏ qua."""
        pass

    def decay_epsilon(self):
        pass

    def reset(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def __repr__(self):
        return "HeuristicAgent(goal-directed + obstacle-avoid)"
