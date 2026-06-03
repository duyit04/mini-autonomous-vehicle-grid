"""
plots.py
--------
Vẽ đồ thị học (learning curves) và so sánh agents.

Hàm chính:
  plot_learning_curves(...)   – đường reward theo episode (rolling average)
  plot_success_rate(...)       – success rate theo episode
  plot_comparison_bar(...)     – bảng bar chart so sánh agents
  plot_policy_heatmap(...)     – heatmap giá trị V(s)
  plot_all(...)                – tạo figure tổng hợp, lưu file
"""

import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


AGENT_COLORS = {
    "random":    "#9E9E9E",
    "heuristic": "#FF9800",
    "q_learning":"#2196F3",
    "sarsa":     "#4CAF50",
}

AGENT_LABELS = {
    "random":    "Random",
    "heuristic": "Heuristic",
    "q_learning":"Q-Learning",
    "sarsa":     "SARSA",
}


# ======================================================================= #
#  Tiện ích                                                                 #
# ======================================================================= #

def rolling_mean(data: list, window: int = 100) -> np.ndarray:
    """Tính rolling average."""
    arr = np.array(data, dtype=float)
    result = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = np.mean(arr[start : i + 1])
    return result


def load_metrics(save_dir: str, agent_name: str, seed: int) -> dict:
    path = os.path.join(save_dir, f"metrics_{agent_name}_seed{seed}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_seeds(save_dir: str, agent_name: str, seeds: list) -> dict:
    """
    Tải metrics cho tất cả seeds và tính mean±std theo episode.
    Returns dict với 'reward_mean', 'reward_std', 'sr_mean', 'sr_std'.
    """
    all_rewards = []
    all_sr      = []

    for seed in seeds:
        m = load_metrics(save_dir, agent_name, seed)
        if not m:
            continue
        all_rewards.append(rolling_mean(m["episode_rewards"], 100))
        all_sr.append(rolling_mean(m["episode_successes"], 100))

    if not all_rewards:
        return {}

    # Pad shorter arrays
    max_len = max(len(a) for a in all_rewards)
    def pad(arr):
        p = np.full(max_len, np.nan)
        p[:len(arr)] = arr
        return p

    all_rewards = np.stack([pad(a) for a in all_rewards])
    all_sr      = np.stack([pad(a) for a in all_sr])

    return {
        "reward_mean": np.nanmean(all_rewards, axis=0),
        "reward_std":  np.nanstd(all_rewards,  axis=0),
        "sr_mean":     np.nanmean(all_sr,       axis=0),
        "sr_std":      np.nanstd(all_sr,        axis=0),
        "n_seeds":     all_rewards.shape[0],
    }


# ======================================================================= #
#  Learning Curves                                                          #
# ======================================================================= #

def plot_learning_curves(
    save_dir: str,
    seeds: list,
    agents: list = None,
    window: int = 100,
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Vẽ đường reward trung bình (rolling) theo episode cho nhiều agents.
    Vùng bóng = ±1 std.
    """
    if agents is None:
        agents = ["random", "heuristic", "q_learning", "sarsa"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    for name in agents:
        data = load_all_seeds(save_dir, name, seeds)
        if not data:
            continue
        n  = len(data["reward_mean"])
        xs = np.arange(n)
        color = AGENT_COLORS.get(name, "black")
        label = AGENT_LABELS.get(name, name)

        ax.plot(xs, data["reward_mean"], color=color, label=label, linewidth=2)
        ax.fill_between(
            xs,
            data["reward_mean"] - data["reward_std"],
            data["reward_mean"] + data["reward_std"],
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(f"Reward (rolling avg {window})", fontsize=11)
    ax.set_title("Learning Curves – Reward", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"  → Lưu: {save_path}")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_success_rate(
    save_dir: str,
    seeds: list,
    agents: list = None,
    ax: plt.Axes = None,
    save_path: str = None,
):
    """Vẽ success rate theo episode."""
    if agents is None:
        agents = ["random", "heuristic", "q_learning", "sarsa"]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    for name in agents:
        data = load_all_seeds(save_dir, name, seeds)
        if not data:
            continue
        n  = len(data["sr_mean"])
        xs = np.arange(n)
        color = AGENT_COLORS.get(name, "black")
        label = AGENT_LABELS.get(name, name)

        ax.plot(xs, data["sr_mean"], color=color, label=label, linewidth=2)
        ax.fill_between(
            xs,
            np.clip(data["sr_mean"] - data["sr_std"], 0, 1),
            np.clip(data["sr_mean"] + data["sr_std"], 0, 1),
            color=color, alpha=0.15,
        )

    ax.axhline(0.85, color="red", linewidth=1.5, linestyle="--",
               label="Target 85%")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Success Rate (rolling avg 100)", fontsize=11)
    ax.set_title("Learning Curves – Success Rate", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"  → Lưu: {save_path}")

    if standalone:
        plt.tight_layout()
        plt.show()


# ======================================================================= #
#  Bar Chart So Sánh                                                        #
# ======================================================================= #

def plot_comparison_bar(
    eval_summary_path: str,
    ax: plt.Axes = None,
    save_path: str = None,
):
    """
    Đọc eval_summary_all.json và vẽ bar chart so sánh.
    """
    if not os.path.exists(eval_summary_path):
        print(f"[WARN] Chưa có file: {eval_summary_path}")
        return

    with open(eval_summary_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)
    summaries = [s for s in summaries if s]

    standalone = ax is None
    if standalone:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    else:
        axes = [ax]

    metrics = [
        ("success_mean", "success_std",   "Success Rate", "%"),
        ("steps_mean",   "steps_std",     "Avg Steps",    ""),
        ("collision_mean","collision_std","Collision Rate","%"),
    ]

    for i, (mean_key, std_key, label, unit) in enumerate(metrics):
        ax_i = axes[i] if standalone else ax
        names  = [AGENT_LABELS.get(s["agent"], s["agent"]) for s in summaries]
        means  = [s[mean_key]  for s in summaries]
        stds   = [s[std_key]   for s in summaries]
        colors = [AGENT_COLORS.get(s["agent"], "gray") for s in summaries]

        bars = ax_i.bar(names, means, yerr=stds, capsize=5,
                        color=colors, alpha=0.85, edgecolor="black")

        if unit == "%":
            ax_i.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(1.0))
        ax_i.set_title(label, fontsize=11)
        ax_i.set_ylabel(f"{label} {unit}", fontsize=9)
        ax_i.grid(axis="y", alpha=0.3)

        # Giá trị trên mỗi bar
        for bar, mean in zip(bars, means):
            val_str = f"{mean:.1%}" if unit == "%" else f"{mean:.1f}"
            ax_i.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.1,
                val_str,
                ha="center", va="bottom", fontsize=8,
            )

    if standalone:
        plt.suptitle("So sánh Agents – Đánh giá 10 Seed", fontsize=13,
                     fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  → Lưu: {save_path}")
        plt.show()


# ======================================================================= #
#  Policy Heatmap                                                           #
# ======================================================================= #

def plot_policy_heatmap(
    env,
    Q_table: np.ndarray,
    goal_id: int = 0,
    heading: int = 0,
    ax: plt.Axes = None,
    save_path: str = None,
    title: str = "Value Heatmap",
):
    """
    Heatmap giá trị V(s) = max_a Q(s,a) trên grid (fix heading & goal_id).
    """
    from envs.custom_env import DirectionalCarEnv

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))

    G = env.GRID_SIZE
    V_grid = np.full((G, G), np.nan)

    for r in range(G):
        for c in range(G):
            if (r, c) not in env.OBSTACLES:
                s = env.state_encoder((r, c, heading, goal_id))
                V_grid[r, c] = float(np.max(Q_table[s]))

    im = ax.imshow(V_grid, cmap="RdYlGn", origin="upper",
                   aspect="equal", interpolation="nearest")

    for r in range(G):
        for c in range(G):
            if (r, c) in env.OBSTACLES:
                ax.add_patch(matplotlib.patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color="#555555"))
            elif not np.isnan(V_grid[r, c]):
                ax.text(c, r, f"{V_grid[r,c]:.0f}",
                        ha="center", va="center",
                        fontsize=7, color="black")

    # Goal markers
    for i, (gr, gc) in enumerate(env.GOALS):
        marker = "★" if i == goal_id else "☆"
        ax.text(gc, gr, marker, ha="center", va="center",
                fontsize=14, color="white" if i == goal_id else "gray")

    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.set_title(
        f"{title}\n(heading={env.HEADING_NAMES[heading]}, goal=G{goal_id})",
        fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8, label="V(s)")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ======================================================================= #
#  Tổng hợp tất cả hình                                                    #
# ======================================================================= #

def plot_all(save_dir: str, seeds: list, figures_dir: str):
    """Tạo và lưu tất cả các hình."""
    os.makedirs(figures_dir, exist_ok=True)

    plot_learning_curves(
        save_dir, seeds,
        save_path=os.path.join(figures_dir, "learning_curves_reward.png"),
    )
    plot_success_rate(
        save_dir, seeds,
        save_path=os.path.join(figures_dir, "learning_curves_sr.png"),
    )
    eval_path = os.path.join(save_dir, "eval_summary_all.json")
    if os.path.exists(eval_path):
        plot_comparison_bar(
            eval_path,
            save_path=os.path.join(figures_dir, "comparison_bar.png"),
        )
    print(f"\n Tất cả hình đã lưu vào: {figures_dir}")


# ======================================================================= #
#  CLI                                                                      #
# ======================================================================= #

if __name__ == "__main__":
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "experiments", "configs.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    save_dir    = os.path.join(root, cfg["logging"]["save_dir"])
    figures_dir = os.path.join(root, cfg["logging"]["plot_dir"])
    seeds       = cfg["evaluation"]["seeds"]

    plot_all(save_dir, seeds, figures_dir)
