"""
agents/sarsa.py
---------------
Expected SARSA Agent.

Công thức:
    Q(s,a) <- Q(s,a) + alpha * [
        r + gamma * sum_a pi(a|s') * Q(s',a) - Q(s,a)
    ]

Khác SARSA thường:
    - SARSA dùng Q(s', a') với a' là action thật sự được chọn.
    - Expected SARSA dùng kỳ vọng trên toàn bộ action tại s'.
"""

import numpy as np


class SARSAAgent:
    name = "sarsa"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    @property
    def Q(self) -> np.ndarray:
        return self.q_table

    @Q.setter
    def Q(self, value: np.ndarray):
        self.q_table = value

    def _greedy_action(self, state: int) -> int:
        """
        Chọn action có Q-value lớn nhất.
        Nếu nhiều action bằng nhau, chọn ngẫu nhiên trong các action tốt nhất.
        """
        q_values = self.q_table[state]
        max_q = q_values.max()
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))

        if best_actions.size == 1:
            return int(best_actions[0])

        return int(self.rng.choice(best_actions))

    def _epsilon_greedy_probs(self, state: int) -> np.ndarray:
        """
        Tính xác suất pi(a|s) theo epsilon-greedy policy hiện tại.

        Với n action:
            Mỗi action nhận epsilon / n.
            Các action greedy nhận thêm (1 - epsilon) chia đều nếu bị tie.
        """
        q_values = self.q_table[state]
        max_q = q_values.max()
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))

        probs = np.full(self.n_actions, self.epsilon / self.n_actions)

        greedy_bonus = (1.0 - self.epsilon) / len(best_actions)
        probs[best_actions] += greedy_bonus

        return probs

    def select_action(self, state: int, eval_mode: bool = False) -> int:
        """
        Chọn action theo epsilon-greedy.

        eval_mode=True:
            epsilon = 0, chỉ chọn greedy action.
        """
        if not eval_mode and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

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
        Cập nhật Q-table theo Expected SARSA.

        Nếu terminated=True:
            target = reward

        Nếu chưa kết thúc:
            expected_next = sum_a pi(a|next_state) * Q(next_state, a)
            target = reward + gamma * expected_next
        """
        if terminated:
            expected_next = 0.0
        else:
            action_probs = self._epsilon_greedy_probs(next_state)
            expected_next = float(np.dot(action_probs, self.q_table[next_state]))

        td_target = reward + self.gamma * expected_next
        td_error = td_target - self.q_table[state, action]

        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset(self):
        self.epsilon = self.epsilon_start

    def save(self, path: str):
        np.save(path, self.q_table)

    def load(self, path: str):
        self.q_table = np.load(path)
