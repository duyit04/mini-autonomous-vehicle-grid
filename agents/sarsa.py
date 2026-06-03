"""
sarsa.py
--------
Thuật toán SARSA (on-policy TD control) với epsilon-greedy.

Khác với Q-Learning (off-policy), SARSA dùng hành động THỰC TẾ kế tiếp
mà agent sẽ thực hiện (a') thay vì hành động tối ưu (max Q):

    Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]

    TD target = r + γ·Q(s',a')   (nếu s' không terminal)
    TD target = r                  (nếu s' là terminal)

Vì phụ thuộc vào a', SARSA phải chọn a' TRƯỚC KHI gọi update().
Xem experiments/train.py để biết cách dùng đúng.
"""

import numpy as np
from typing import Optional


class SARSAAgent:
    """
    SARSA Agent (on-policy TD control).

    Parameters
    ----------
    n_states : int
    n_actions : int
    alpha : float           Learning rate α
    gamma : float           Discount factor γ
    epsilon_start : float   Epsilon ban đầu
    epsilon_end : float     Epsilon tối thiểu
    epsilon_decay : float   Hệ số suy giảm mỗi episode
    seed : int, optional
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
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.name          = "SARSA"

        self.rng = np.random.RandomState(seed)
        self.Q   = np.zeros((n_states, n_actions), dtype=np.float64)
        self.total_updates = 0

    # ------------------------------------------------------------------ #
    #  Chọn hành động (epsilon-greedy, giống Q-Learning)                   #
    # ------------------------------------------------------------------ #

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        """
        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.randint(self.n_actions))
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------ #
    #  Cập nhật Q-table (SARSA cần next_action)                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        terminated: bool,
        next_action: Optional[int] = None,
        **kwargs,
    ) -> float:
        """
        Cập nhật Q(s, a) theo quy tắc SARSA.

        QUAN TRỌNG: next_action phải được truyền vào (không phải None).
        Nếu terminated, next_action bị bỏ qua.

        Công thức:
            TD_target = r                         nếu terminated
            TD_target = r + γ·Q(s',a')            nếu không terminated
            TD_error  = TD_target − Q(s,a)
            Q(s,a)   += α · TD_error

        Returns
        -------
        float
            TD error.
        """
        if terminated:
            td_target = reward
        else:
            assert next_action is not None, (
                "SARSA cần next_action khi trạng thái không phải terminal. "
                "Xem experiments/train.py."
            )
            td_target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        self.total_updates += 1

        return float(td_error)

    # ------------------------------------------------------------------ #
    #  Epsilon decay                                                       #
    # ------------------------------------------------------------------ #

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Lưu / tải                                                           #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        np.save(path, self.Q)

    def load(self, path: str):
        self.Q = np.load(path)
        assert self.Q.shape == (self.n_states, self.n_actions)

    # ------------------------------------------------------------------ #
    #  Tiện ích                                                            #
    # ------------------------------------------------------------------ #

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)

    def reset(self):
        self.Q[:] = 0.0
        self.epsilon = 1.0
        self.total_updates = 0

    def __repr__(self):
        return (
            f"SARSAAgent("
            f"α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, "
            f"updates={self.total_updates})"
        )
