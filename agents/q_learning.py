"""
q_learning.py
-------------
Thuật toán Q-Learning với epsilon-greedy exploration.

Cập nhật Q-table:
    Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]

Trong đó:
    α   : learning rate (tốc độ học)
    γ   : discount factor (hệ số chiết khấu)
    r   : reward tức thời
    TD target = r + γ·max_a' Q(s',a')   (nếu s' không phải terminal)
    TD target = r                         (nếu s' là terminal)
    TD error  = TD_target − Q(s,a)

Epsilon-greedy:
    - Với xác suất ε  : chọn hành động ngẫu nhiên (exploration)
    - Với xác suất 1-ε: chọn argmax Q(s,·)        (exploitation)
    - ε suy giảm mỗi episode: ε ← max(ε_end, ε · ε_decay)
"""

import numpy as np
from typing import Optional


class QLearningAgent:
    """
    Q-Learning Agent (off-policy TD control).

    Parameters
    ----------
    n_states : int
        Số trạng thái (kích thước Q-table chiều 0).
    n_actions : int
        Số hành động (kích thước Q-table chiều 1).
    alpha : float
        Learning rate α ∈ (0, 1].
    gamma : float
        Discount factor γ ∈ [0, 1].
    epsilon_start : float
        Epsilon ban đầu (xác suất khám phá).
    epsilon_end : float
        Epsilon tối thiểu sau khi suy giảm.
    epsilon_decay : float
        Hệ số suy giảm epsilon mỗi episode.
    seed : int, optional
        Hạt ngẫu nhiên.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float         = 0.1,
        gamma: float         = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float   = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int]  = None,
    ):
        self.n_states     = n_states
        self.n_actions    = n_actions
        self.alpha        = alpha
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.name         = "Q-Learning"

        self.rng = np.random.RandomState(seed)

        # Q-table khởi tạo về 0
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

        # Thống kê
        self.total_updates = 0

    # ------------------------------------------------------------------ #
    #  Chọn hành động                                                      #
    # ------------------------------------------------------------------ #

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : int
            Trạng thái hiện tại (đã mã hóa).
        eval_mode : bool
            Nếu True → epsilon = 0 (greedy hoàn toàn).

        Returns
        -------
        int
            Hành động được chọn.
        """
        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.randint(self.n_actions))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------ #
    #  Cập nhật Q-table                                                    #
    # ------------------------------------------------------------------ #

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        **kwargs,
    ) -> float:
        """
        Cập nhật Q(s, a) theo quy tắc Q-Learning.

        Công thức:
            TD_target = r                         nếu terminated
            TD_target = r + γ·max_a' Q(s',a')    nếu không terminated
            TD_error  = TD_target − Q(s,a)
            Q(s,a)   += α · TD_error

        Returns
        -------
        float
            TD error (để logging/debugging).
        """
        if terminated:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        self.total_updates += 1

        return float(td_error)

    # ------------------------------------------------------------------ #
    #  Epsilon decay                                                       #
    # ------------------------------------------------------------------ #

    def decay_epsilon(self):
        """Suy giảm epsilon sau mỗi episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Lưu / tải Q-table                                                   #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Lưu Q-table ra file .npy."""
        np.save(path, self.Q)

    def load(self, path: str):
        """Tải Q-table từ file .npy."""
        self.Q = np.load(path)
        assert self.Q.shape == (self.n_states, self.n_actions), (
            f"Kích thước Q-table không khớp: "
            f"expect {(self.n_states, self.n_actions)}, got {self.Q.shape}"
        )

    # ------------------------------------------------------------------ #
    #  Tiện ích                                                            #
    # ------------------------------------------------------------------ #

    def get_policy(self) -> np.ndarray:
        """Trả về mảng policy: policy[s] = argmax_a Q(s,a)."""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        """Trả về mảng V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)

    def reset(self):
        """Reset Q-table về 0 và epsilon về epsilon_start."""
        self.Q[:] = 0.0
        self.epsilon = 1.0
        self.total_updates = 0

    def __repr__(self):
        return (
            f"QLearningAgent("
            f"α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, "
            f"updates={self.total_updates})"
        )
