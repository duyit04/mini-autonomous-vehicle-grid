"""
agents/random_agent.py
----------------------
[THÀNH VIÊN 1] Cài đặt RandomAgent: chọn action ngẫu nhiên đều.

Thuật toán:
    Mỗi bước chọn một action ngẫu nhiên trong {0, 1, ..., n_actions-1}
    với xác suất đều nhau. Không có bộ nhớ, không có học.

Chạy test:
    python -m pytest tests/test_agents.py::TestRandomAgent -v
"""

import numpy as np


class RandomAgent:
    """
    Agent chọn action ngẫu nhiên đều.

    Parameters
    ----------
    n_actions : int   Số lượng action hợp lệ (môi trường: 4)
    seed      : int   Seed cho numpy RNG (để test reproducible)
    """

    name = "random"

    def __init__(self, n_actions: int, seed: int = None):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, eval_mode: bool = False, **kwargs) -> int:
        """
        Trả về một action ngẫu nhiên.

        Parameters
        ----------
        state : int   Trạng thái hiện tại (bỏ qua, không dùng)
        eval_mode : bool   Chế độ đánh giá (bỏ qua)

        Returns
        -------
        int   Action trong [0, n_actions)
        """
        return int(self.rng.integers(self.n_actions))

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

