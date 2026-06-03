"""
random_agent.py
---------------
Agent chọn hành động ngẫu nhiên đều (uniform random).
Dùng làm baseline thấp nhất để so sánh.
"""

import numpy as np
from typing import Optional


class RandomAgent:
    """
    Baseline: chọn hành động ngẫu nhiên, không học gì.

    Parameters
    ----------
    n_actions : int
        Số hành động hợp lệ.
    seed : int, optional
        Hạt ngẫu nhiên (để tái lập kết quả đánh giá).
    """

    def __init__(self, n_actions: int, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
        self.name = "Random"

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Chọn hành động ngẫu nhiên đều, bất kể state và eval_mode.

        Parameters
        ----------
        state : int
            Trạng thái hiện tại (bỏ qua).
        eval_mode : bool
            Bỏ qua – RandomAgent luôn ngẫu nhiên.

        Returns
        -------
        int
            Hành động trong [0, n_actions).
        """
        return int(self.rng.randint(self.n_actions))

    def update(self, state, action, reward, next_state, terminated, **kwargs):
        """Không có cơ chế học – bỏ qua."""
        pass

    def decay_epsilon(self):
        """Không có epsilon – bỏ qua."""
        pass

    def reset(self):
        """Không có trạng thái nội tại cần reset."""
        pass

    def save(self, path: str):
        """Không có gì để lưu."""
        pass

    def load(self, path: str):
        """Không có gì để tải."""
        pass

    def __repr__(self):
        return f"RandomAgent(n_actions={self.n_actions})"
