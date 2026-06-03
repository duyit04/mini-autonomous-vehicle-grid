"""
render.py
---------
Vẽ grid 7×7 bằng matplotlib với:
  - Xe có mũi tên thể hiện hướng (heading)
  - Obstacle tô xám đậm
  - Goal tô xanh lá (active) hoặc xanh nhạt (inactive)
  - Ô trống màu trắng
  - Đường đi của agent (replay trail)
"""

import os
import sys

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.custom_env import DirectionalCarEnv


# ======================================================================= #
#  Hằng số màu sắc                                                         #
# ======================================================================= #

COLOR_EMPTY    = "#FFFFFF"
COLOR_OBSTACLE = "#555555"
COLOR_GOAL_ACT = "#4CAF50"   # goal đang hướng đến
COLOR_GOAL_INK = "#C8E6C9"   # goal khác (inactive)
COLOR_CAR      = "#E53935"   # xe
COLOR_TRAIL    = "#90CAF9"   # đường đi
COLOR_GRID     = "#CCCCCC"   # viền ô

# Mũi tên heading (góc quay tính từ trục x dương, ngược chiều kim đồng hồ)
HEADING_ANGLES = {
    DirectionalCarEnv.NORTH: 90,   # ↑
    DirectionalCarEnv.EAST:  0,    # →
    DirectionalCarEnv.SOUTH: 270,  # ↓
    DirectionalCarEnv.WEST:  180,  # ←
}


# ======================================================================= #
#  Hàm vẽ grid                                                             #
# ======================================================================= #

def draw_grid(
    ax: plt.Axes,
    env: DirectionalCarEnv,
    trail: list = None,
    title: str = "",
    show_coords: bool = True,
) -> None:
    """
    Vẽ trạng thái hiện tại của môi trường lên ax.

    Parameters
    ----------
    ax      : matplotlib Axes
    env     : DirectionalCarEnv (đã gọi reset() )
    trail   : list of (r,c) – đường đi lưu lại từ đầu episode
    title   : tiêu đề phụ
    show_coords : có hiển thị tọa độ không
    """
    ax.clear()
    G = env.GRID_SIZE
    state = env.state

    if state is None:
        ax.set_title("Chưa khởi tạo – gọi reset()")
        return

    car_r, car_c, heading, goal_id = state
    goal_r, goal_c = env.GOALS[goal_id]

    # --- Vẽ nền ô ---
    for r in range(G):
        for c in range(G):
            if (r, c) in env.OBSTACLES:
                color = COLOR_OBSTACLE
            elif (r, c) == env.GOALS[goal_id]:
                color = COLOR_GOAL_ACT
            elif (r, c) in env.GOALS:
                color = COLOR_GOAL_INK
            elif trail and (r, c) in trail:
                color = COLOR_TRAIL
            else:
                color = COLOR_EMPTY

            rect = mpatches.FancyBboxPatch(
                (c, G - 1 - r),
                1, 1,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor=COLOR_GRID,
                linewidth=0.8,
            )
            ax.add_patch(rect)

    # --- Nhãn goal ---
    for i, (gr, gc) in enumerate(env.GOALS):
        ax.text(
            gc + 0.5, G - 1 - gr + 0.15,
            f"G{i}",
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="white" if i == goal_id else "#2E7D32",
        )

    # --- Đường đi (trail) ---
    if trail and len(trail) > 1:
        xs = [c + 0.5 for (r, c) in trail]
        ys = [G - 1 - r + 0.5 for (r, c) in trail]
        ax.plot(xs, ys, color="#1565C0", linewidth=1.5,
                linestyle="--", alpha=0.6, zorder=2)

    # --- Xe ---
    _draw_car(ax, car_r, car_c, heading, G)

    # --- Obstacle labels ---
    for (or_, oc) in env.OBSTACLES:
        ax.text(oc + 0.5, G - 1 - or_ + 0.5, "✕",
                ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")

    # --- Axes ---
    ax.set_xlim(0, G)
    ax.set_ylim(0, G)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0.5, G, 1))
    ax.set_yticks(np.arange(0.5, G, 1))
    ax.set_xticklabels(range(G) if show_coords else [])
    ax.set_yticklabels(list(reversed(range(G))) if show_coords else [])
    ax.tick_params(length=0)

    dist = abs(car_r - goal_r) + abs(car_c - goal_c)
    full_title = (
        f"{title}\n"
        f"State: ({car_r},{car_c},{env.HEADING_NAMES[heading]},G{goal_id})  "
        f"dist={dist}  step={env.steps}"
    )
    ax.set_title(full_title, fontsize=9, pad=6)


def _draw_car(ax, r, c, heading, G):
    """Vẽ hình xe (vòng tròn + mũi tên hướng)."""
    cx = c + 0.5
    cy = G - 1 - r + 0.5

    # Thân xe
    circle = plt.Circle((cx, cy), 0.30,
                         facecolor=COLOR_CAR, edgecolor="#B71C1C",
                         linewidth=1.5, zorder=5)
    ax.add_patch(circle)

    # Mũi tên hướng
    angle = HEADING_ANGLES[heading]
    dx = 0.35 * np.cos(np.radians(angle))
    dy = 0.35 * np.sin(np.radians(angle))
    ax.annotate(
        "",
        xy=(cx + dx, cy + dy),
        xytext=(cx - dx * 0.3, cy - dy * 0.3),
        arrowprops=dict(arrowstyle="-|>", color="white",
                        lw=2.0, mutation_scale=14),
        zorder=6,
    )


# ======================================================================= #
#  Vẽ policy (mũi tên trên grid)                                           #
# ======================================================================= #

def draw_policy(
    ax: plt.Axes,
    env: DirectionalCarEnv,
    Q_table: np.ndarray,
    goal_id: int = 0,
    heading: int = DirectionalCarEnv.NORTH,
    title: str = "Policy",
) -> None:
    """
    Vẽ policy dạng mũi tên tại mỗi ô (fixe heading + goal_id).

    Parameters
    ----------
    Q_table : (n_states, 4) array
    goal_id : int   goal cố định khi hiển thị policy
    heading : int   heading cố định khi hiển thị policy
    """
    ax.clear()
    G = env.GRID_SIZE

    ACTION_ARROWS = {
        env.FORWARD:    ("↑", HEADING_ANGLES[heading]),
        env.TURN_LEFT:  ("↺", 0),
        env.TURN_RIGHT: ("↻", 0),
        env.STOP:       ("■", 0),
    }

    for r in range(G):
        for c in range(G):
            if (r, c) in env.OBSTACLES:
                color = COLOR_OBSTACLE
            elif (r, c) == env.GOALS[goal_id]:
                color = COLOR_GOAL_ACT
            elif (r, c) in env.GOALS:
                color = COLOR_GOAL_INK
            else:
                color = COLOR_EMPTY

            rect = mpatches.FancyBboxPatch(
                (c, G - 1 - r), 1, 1,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor=COLOR_GRID,
                linewidth=0.8,
            )
            ax.add_patch(rect)

            if (r, c) not in env.OBSTACLES:
                state = env.state_encoder((r, c, heading, goal_id))
                best_a = int(np.argmax(Q_table[state]))
                q_val  = float(Q_table[state, best_a])

                # Mũi tên/ký hiệu
                sym = _action_to_arrow(best_a, heading)
                ax.text(
                    c + 0.5, G - 1 - r + 0.55,
                    sym,
                    ha="center", va="center",
                    fontsize=14, color="#1A237E",
                    fontweight="bold",
                )
                # Giá trị Q nhỏ
                ax.text(
                    c + 0.5, G - 1 - r + 0.18,
                    f"{q_val:.0f}",
                    ha="center", va="center",
                    fontsize=6, color="#555555",
                )

    ax.set_xlim(0, G)
    ax.set_ylim(0, G)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0.5, G, 1))
    ax.set_yticks(np.arange(0.5, G, 1))
    ax.set_xticklabels(range(G))
    ax.set_yticklabels(list(reversed(range(G))))
    ax.tick_params(length=0)
    ax.set_title(f"{title}\n(heading={env.HEADING_NAMES[heading]}, goal=G{goal_id})",
                 fontsize=9, pad=6)


def _action_to_arrow(action: int, heading: int) -> str:
    """Trả về ký tự unicode biểu diễn hành động theo heading hiện tại."""
    # heading-relative arrows for FORWARD
    fwd_arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    if action == DirectionalCarEnv.FORWARD:
        return fwd_arrows[heading]
    elif action == DirectionalCarEnv.TURN_LEFT:
        return "↺"
    elif action == DirectionalCarEnv.TURN_RIGHT:
        return "↻"
    elif action == DirectionalCarEnv.STOP:
        return "■"
    return "?"


# ======================================================================= #
#  Demo độc lập                                                             #
# ======================================================================= #

if __name__ == "__main__":
    env = DirectionalCarEnv()
    obs, _ = env.reset(seed=42)

    fig, ax = plt.subplots(figsize=(5, 5))
    trail = []
    r, c, h, gid = env.state_decoder(obs)
    trail.append((r, c))

    draw_grid(ax, env, trail=trail, title="Demo render.py")
    plt.tight_layout()
    plt.show()
