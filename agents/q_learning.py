"""
agents/q_learning.py
--------------------
[THÀNH VIÊN 3] Cài đặt QLearningAgent: thuật toán Q-Learning (off-policy TD).

Thuật toán:
    Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

    - Off-policy: cập nhật dùng max Q(s',a') bất kể action thực sự chọn
    - Epsilon-greedy: với xác suất ε chọn ngẫu nhiên, còn lại chọn argmax Q
    - Epsilon decay: ε = max(epsilon_end, ε × epsilon_decay) sau mỗi episode

Chạy test:
    python -m pytest tests/test_agents.py::TestQLearningAgent -v
"""

import numpy as np


class QLearningAgent:
    """
    Q-Learning Agent (off-policy TD).

    Parameters
    ----------
    n_states      : int    Số trạng thái (môi trường: 588)
    n_actions     : int    Số hành động (môi trường: 4)
    alpha         : float  Learning rate (e.g. 0.1)
    gamma         : float  Discount factor (e.g. 0.99)
    epsilon_start : float  Epsilon ban đầu (e.g. 1.0)
    epsilon_end   : float  Epsilon nhỏ nhất (e.g. 0.01)
    epsilon_decay : float  Hệ số giảm epsilon mỗi episode (e.g. 0.995)
    seed          : int    Seed cho numpy RNG
    """

    name = "q_learning"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None,
    ):
        # TODO: lưu tất cả hyperparameter, khởi tạo Q-table (n_states × n_actions)
        #       với giá trị 0, khởi tạo numpy RNG với seed
        raise NotImplementedError

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Chọn action theo epsilon-greedy.

        Parameters
        ----------
        state     : int    State hiện tại (index)
        eval_mode : bool   Nếu True → greedy thuần (không random)

        Returns
        -------
        int   Action trong [0, n_actions)
        """
        # TODO:
        #   - eval_mode=True  → luôn chọn argmax Q[state]
        #   - eval_mode=False → với xác suất ε chọn ngẫu nhiên,
        #                        còn lại chọn argmax Q[state]
        raise NotImplementedError

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        **kwargs,
    ):
        """
        Cập nhật Q-table theo công thức Q-Learning.

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

        Lưu ý: nếu terminated=True thì không có Q(s') → target = r
        """
        # TODO: tính td_target, td_error, cập nhật Q[state][action]
        raise NotImplementedError

    def decay_epsilon(self):
        """Giảm epsilon sau mỗi episode: ε = max(epsilon_end, ε × epsilon_decay)"""
        # TODO
        raise NotImplementedError

    def reset(self):
        """Reset epsilon về epsilon_start (dùng khi train lại từ đầu)."""
        # TODO
        raise NotImplementedError

    def save(self, path: str):
        """Lưu Q-table ra file .npy"""
        # TODO: np.save(path, self.q_table)
        raise NotImplementedError

    def load(self, path: str):
        """Tải Q-table từ file .npy"""
        # TODO: self.q_table = np.load(path)
        raise NotImplementedError
