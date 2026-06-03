"""
custom_env.py
-------------
Môi trường RL chính: Xe tự hành mini trong grid 7×7 có hướng quay.

MDP:
  State   : (row, col, heading, goal_id)   →  588 trạng thái
  Action  : forward / turn_left / turn_right / stop   (4 hành động)
  Reward  : xem bảng reward bên dưới
  Terminal: va chạm obstacle/tường, stop đúng goal, hết max_steps

Bản đồ (7×7):
  Hàng/Cột  0    1    2    3    4    5    6
       0     .    .    .    .    .    .   G0
       1     .    X    .    .    X    .    .
       2     .    .    .    .    X    .    .
       3     .    .    X    .    .    .   G1
       4     .    .    .    .    X    .    .
       5     .    X    .    .    .    X    .
       6    G2    .    .    .    .    .    .

  X  = obstacle
  G0 = goal 0 tại (0,6)
  G1 = goal 1 tại (3,6)
  G2 = goal 2 tại (6,0)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from envs.base_env import BaseEnv


class DirectionalCarEnv(BaseEnv):
    """
    Môi trường xe tự hành mini trong grid 7×7.

    Heading:
        NORTH=0  →  row-1
        EAST =1  →  col+1
        SOUTH=2  →  row+1
        WEST =3  →  col-1

    TURN_LEFT  : heading = (heading - 1) % 4
    TURN_RIGHT : heading = (heading + 1) % 4
    """

    # ------------------------------------------------------------------ #
    #  Hằng số                                                             #
    # ------------------------------------------------------------------ #

    GRID_SIZE = 7

    # Heading
    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
    HEADING_NAMES  = {0: "NORTH", 1: "EAST", 2: "SOUTH", 3: "WEST"}
    HEADING_ARROWS = {0: "↑",     1: "→",    2: "↓",     3: "←"}

    # Action
    FORWARD, TURN_LEFT, TURN_RIGHT, STOP = 0, 1, 2, 3
    ACTION_NAMES = {0: "forward", 1: "turn_left", 2: "turn_right", 3: "stop"}

    # Delta (dr, dc) cho mỗi hướng
    DELTAS: Dict[int, Tuple[int, int]] = {
        0: (-1,  0),   # NORTH
        1: ( 0,  1),   # EAST
        2: ( 1,  0),   # SOUTH
        3: ( 0, -1),   # WEST
    }

    # Obstacle cố định
    OBSTACLES = frozenset([
        (1, 1), (1, 4),
        (2, 4),
        (3, 2),
        (4, 4),
        (5, 1), (5, 5),
    ])

    # Goal cố định: goal_id → (row, col)
    GOALS: List[Tuple[int, int]] = [(0, 6), (3, 6), (6, 0)]

    # Reward
    R_STEP          = -1    # mỗi bước
    R_CLOSER        = +1    # tiến gần goal
    R_FARTHER       = -1    # tiến xa goal
    R_COLLISION     = -30   # va chạm obstacle / tường
    R_GOAL          = +50   # stop đúng tại goal
    R_WRONG_STOP    = -10   # stop sai vị trí

    # ------------------------------------------------------------------ #
    #  Khởi tạo                                                            #
    # ------------------------------------------------------------------ #

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self._n_states  = self.GRID_SIZE * self.GRID_SIZE * 4 * len(self.GOALS)
        self._n_actions = 4

        # Vị trí xuất phát hợp lệ (không phải obstacle, không phải goal)
        _goals_set = set(self.GOALS)
        self.valid_starts: List[Tuple[int, int]] = [
            (r, c)
            for r in range(self.GRID_SIZE)
            for c in range(self.GRID_SIZE)
            if (r, c) not in self.OBSTACLES and (r, c) not in _goals_set
        ]

        # Trạng thái nội tại
        self._state: Optional[Tuple[int, int, int, int]] = None
        self._steps: int = 0
        self._rng: Optional[np.random.RandomState] = None

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def state(self) -> Optional[Tuple[int, int, int, int]]:
        """Trạng thái hiện tại dạng tuple thô (r, c, heading, goal_id)."""
        return self._state

    @property
    def steps(self) -> int:
        """Số bước đã thực hiện trong episode hiện tại."""
        return self._steps

    # ------------------------------------------------------------------ #
    #  reset()                                                             #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> Tuple[int, Dict]:
        """
        Khởi tạo lại môi trường.

        Vị trí và hướng ban đầu của xe được chọn ngẫu nhiên từ valid_starts.
        Goal cũng được chọn ngẫu nhiên trong [0, 1, 2].

        Parameters
        ----------
        seed : int, optional
            Hạt ngẫu nhiên để tái lập kết quả.

        Returns
        -------
        (encoded_state, info)
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        elif self._rng is None:
            self._rng = np.random.RandomState()

        idx     = self._rng.randint(len(self.valid_starts))
        r, c    = self.valid_starts[idx]
        heading = int(self._rng.randint(4))
        goal_id = int(self._rng.randint(len(self.GOALS)))

        self._state = (r, c, heading, goal_id)
        self._steps = 0

        return self.state_encoder(self._state), {}

    # ------------------------------------------------------------------ #
    #  step()                                                              #
    # ------------------------------------------------------------------ #

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Thực hiện hành động và trả về kết quả.

        Bảng reward:
          FORWARD → gần goal  : R_STEP + R_CLOSER   =  0
          FORWARD → xa  goal  : R_STEP + R_FARTHER  = -2
          FORWARD → cùng dist : R_STEP              = -1
          FORWARD → va chạm   : R_STEP + R_COLLISION = -31, terminal
          TURN_LEFT / TURN_RIGHT : R_STEP           = -1
          STOP đúng goal      : R_GOAL              = +50, terminal
          STOP sai            : R_STEP + R_WRONG_STOP = -11

        Parameters
        ----------
        action : int   (0=forward, 1=turn_left, 2=turn_right, 3=stop)

        Returns
        -------
        (encoded_next_state, reward, terminated, truncated, info)
        """
        assert self._state is not None, "Gọi reset() trước khi step()."
        assert 0 <= action < self._n_actions, (
            f"Hành động không hợp lệ: {action}. Hợp lệ: 0–{self._n_actions - 1}."
        )

        r, c, heading, goal_id = self._state
        goal_r, goal_c = self.GOALS[goal_id]

        prev_dist  = abs(r - goal_r) + abs(c - goal_c)
        terminated = False
        truncated  = False
        info       = {
            "collision":    False,
            "reached_goal": False,
            "action_name":  self.ACTION_NAMES[action],
        }

        # ---------- xử lý từng hành động ----------
        if action == self.FORWARD:
            dr, dc = self.DELTAS[heading]
            nr, nc = r + dr, c + dc

            # Va chạm tường hoặc obstacle → phạt nặng, terminal
            if not (0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE):
                reward              = self.R_STEP + self.R_COLLISION
                info["collision"]   = True
                terminated          = True
                # xe không di chuyển
            elif (nr, nc) in self.OBSTACLES:
                reward              = self.R_STEP + self.R_COLLISION
                info["collision"]   = True
                terminated          = True
            else:
                # Di chuyển hợp lệ
                r, c      = nr, nc
                new_dist  = abs(r - goal_r) + abs(c - goal_c)

                if new_dist < prev_dist:
                    reward = self.R_STEP + self.R_CLOSER
                elif new_dist > prev_dist:
                    reward = self.R_STEP + self.R_FARTHER
                else:
                    reward = self.R_STEP

        elif action == self.TURN_LEFT:
            heading = (heading - 1) % 4
            reward  = self.R_STEP

        elif action == self.TURN_RIGHT:
            heading = (heading + 1) % 4
            reward  = self.R_STEP

        elif action == self.STOP:
            if (r, c) == (goal_r, goal_c):
                reward              = self.R_GOAL
                info["reached_goal"] = True
                terminated          = True
            else:
                reward = self.R_STEP + self.R_WRONG_STOP

        # Cập nhật trạng thái
        self._state  = (r, c, heading, goal_id)
        self._steps += 1

        if not terminated and self._steps >= self.max_steps:
            truncated = True

        info["steps"] = self._steps
        return self.state_encoder(self._state), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  render()                                                            #
    # ------------------------------------------------------------------ #

    def render(self) -> str:
        """
        Trả về biểu diễn văn bản của grid.

        Ký tự:
          ↑/→/↓/←  : xe (hướng hiện tại)
          X         : obstacle
          G         : goal đang hướng đến (active)
          g         : goal khác (inactive)
          .         : ô trống
        """
        if self._state is None:
            return "[Chưa khởi tạo – gọi reset() trước]"

        r, c, heading, goal_id = self._state
        goal_r, goal_c = self.GOALS[goal_id]
        dist = abs(r - goal_r) + abs(c - goal_c)

        # Xây lưới ký tự
        grid = [["." for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        for (or_, oc) in self.OBSTACLES:
            grid[or_][oc] = "X"
        for i, (gr, gc) in enumerate(self.GOALS):
            grid[gr][gc] = "G" if i == goal_id else "g"
        grid[r][c] = self.HEADING_ARROWS[heading]

        # Header
        lines = [
            "=" * 29,
            f" Bước: {self._steps:4d}  |  Goal: G{goal_id}{self.GOALS[goal_id]}",
            f" Vị trí: ({r},{c})  Hướng: {self.HEADING_NAMES[heading]}",
            f" Manhattan → Goal: {dist}",
            "=" * 29,
            "   " + " ".join(str(i) for i in range(self.GRID_SIZE)),
        ]
        for i, row in enumerate(grid):
            lines.append(f"{i}  " + " ".join(row))
        lines.append("=" * 29)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  state_encoder / state_decoder                                       #
    # ------------------------------------------------------------------ #

    def state_encoder(self, state: Tuple[int, int, int, int]) -> int:
        """
        (r, c, heading, goal_id) → chỉ số nguyên trong [0, n_states).

        Công thức:
            idx = r * (G * 4 * N_GOALS)
                + c * (4 * N_GOALS)
                + heading * N_GOALS
                + goal_id
        trong đó G = GRID_SIZE = 7, N_GOALS = 3
        """
        r, c, heading, goal_id = state
        N = len(self.GOALS)       # 3
        H = 4                      # số heading
        G = self.GRID_SIZE         # 7

        return r * (G * H * N) + c * (H * N) + heading * N + goal_id

    def state_decoder(self, encoded: int) -> Tuple[int, int, int, int]:
        """
        Chỉ số nguyên → (r, c, heading, goal_id).

        Nghịch đảo của state_encoder.
        """
        N = len(self.GOALS)
        H = 4
        G = self.GRID_SIZE

        goal_id  = encoded % N;       encoded //= N
        heading  = encoded % H;       encoded //= H
        c        = encoded % G;       encoded //= G
        r        = encoded

        return (int(r), int(c), int(heading), int(goal_id))

    # ------------------------------------------------------------------ #
    #  Tiện ích                                                            #
    # ------------------------------------------------------------------ #

    def get_all_valid_positions(self) -> List[Tuple[int, int]]:
        """Danh sách tất cả ô không phải obstacle."""
        return [
            (r, c)
            for r in range(self.GRID_SIZE)
            for c in range(self.GRID_SIZE)
            if (r, c) not in self.OBSTACLES
        ]

    def manhattan_distance(self, pos: Tuple[int, int], goal_id: int) -> int:
        gr, gc = self.GOALS[goal_id]
        return abs(pos[0] - gr) + abs(pos[1] - gc)

    def is_valid_cell(self, r: int, c: int) -> bool:
        return (0 <= r < self.GRID_SIZE and
                0 <= c < self.GRID_SIZE and
                (r, c) not in self.OBSTACLES)
