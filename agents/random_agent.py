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
        # TODO: lưu n_actions, khởi tạo numpy RNG với seed
        raise NotImplementedError

    def select_action(self, state) -> int:
        """
        Trả về một action ngẫu nhiên.

        Parameters
        ----------
        state : int   Trạng thái hiện tại (bỏ qua, không dùng)

        Returns
        -------
        int   Action trong [0, n_actions)
        """
        # TODO: trả về action ngẫu nhiên đều trong [0, n_actions)
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
