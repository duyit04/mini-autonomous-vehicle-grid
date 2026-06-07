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
        # Lưu hyperparameter
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)

        # RNG tái lập được
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Q-table khởi tạo 0: shape (n_states, n_actions)
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def _greedy_action(self, state: int) -> int:
        """
        Chọn argmax Q[state] với random tie-breaking.

        Khi nhiều action cùng giá trị Q lớn nhất (rất phổ biến lúc đầu khi
        Q-table toàn 0), ta chọn ngẫu nhiên trong số đó thay vì luôn lấy
        index nhỏ nhất → tránh thiên lệch về action 0, học nhanh và đều hơn.
        """
        q_values = self.q_table[state]
        max_q = q_values.max()
        # Các action đạt giá trị lớn nhất (dùng isclose để an toàn với float)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        if best_actions.size == 1:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

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
        # Khám phá: với xác suất ε chọn ngẫu nhiên (chỉ khi training)
        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        # Khai thác: chọn argmax với random tie-breaking
        return self._greedy_action(state)

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
        # Giá trị bootstrap: 0 nếu kết thúc, ngược lại max Q(s', ·)
        best_next = 0.0 if terminated else float(self.q_table[next_state].max())

        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Giảm epsilon sau mỗi episode: ε = max(epsilon_end, ε × epsilon_decay)"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self):
        """Reset epsilon về epsilon_start (dùng khi train lại từ đầu)."""
        self.epsilon = self.epsilon_start

    def save(self, path: str):
        """Lưu Q-table ra file .npy"""
        np.save(path, self.q_table)

    def load(self, path: str):
        """Tải Q-table từ file .npy"""
        self.q_table = np.load(path)
