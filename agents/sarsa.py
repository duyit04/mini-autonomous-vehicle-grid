"""
agents/sarsa.py
---------------
[THÀNH VIÊN 4] Cài đặt SARSAAgent: thuật toán SARSA (on-policy TD).

Thuật toán:
    Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]

    - On-policy: cập nhật dùng Q(s', a') với a' là action THỰC SỰ sẽ chọn
    - Epsilon-greedy: như Q-Learning
    - Khác Q-Learning: target dùng Q(s',a') thay vì max Q(s',a')

Lưu ý quan trọng (train.py đã xử lý hộ):
    SARSA chọn a' TRƯỚC khi gọi update() → next_action được truyền vào
    update() qua tham số next_action. Bạn chỉ cần dùng giá trị đó.

Chạy test:
    python -m pytest tests/test_agents.py::TestSARSAAgent -v
"""

import numpy as np


class SARSAAgent:
    """
    SARSA Agent (on-policy TD).

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

    name = "sarsa"

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
        # TODO: giống QLearningAgent.select_action()
        raise NotImplementedError

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        next_action: int = 0,
        **kwargs,
    ):
        """
        Cập nhật Q-table theo công thức SARSA.

        Q(s,a) ← Q(s,a) + α · [r + γ · Q(s',a') − Q(s,a)]

        Parameters
        ----------
        next_action : int   Action a' đã được chọn cho bước tiếp theo
                            (train.py truyền vào – bạn chỉ cần dùng)

        Lưu ý: nếu terminated=True thì không có Q(s') → target = r
        """
        # TODO: tính td_target dùng Q[next_state][next_action] (không phải max),
        #       tính td_error, cập nhật Q[state][action]
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
