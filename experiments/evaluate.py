"""
evaluate.py
-----------
Đánh giá tất cả agents sau khi huấn luyện.
Chạy 10 seed, mỗi seed 200 episode đánh giá với epsilon = 0 (greedy).

Báo cáo:
  - Mean ± Std của: total reward, steps, success rate, collision rate
  - Bảng so sánh 4 agents
  - Learning curves (lưu hình)

Chạy:
    python experiments/evaluate.py
    python experiments/evaluate.py --agent q_learning
"""

import argparse
import json
import os
import sys

# Fix Unicode encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import DirectionalCarEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent
from experiments.train import (
    make_env,
    run_episode_random_or_heuristic,
    run_episode_qlearning,
    run_episode_sarsa,
)


# ======================================================================= #
#  Hàm đánh giá một agent trên nhiều episode                               #
# ======================================================================= #

def evaluate_agent(agent, env, agent_name: str, n_episodes: int,
                   seed_offset: int = 0) -> dict:
    """
    Chạy agent ở chế độ đánh giá (eval_mode=True, epsilon=0).

    Returns
    -------
    dict với keys: rewards, steps, successes, collisions
    """
    rewards, steps, successes, collisions = [], [], [], []

    for i in range(n_episodes):
        ep_seed = seed_offset * 10000 + i  # seed khác nhau cho mỗi episode

        if agent_name in ("random", "heuristic"):
            result = run_episode_random_or_heuristic(agent, env, seed_ep=ep_seed)
        elif agent_name == "q_learning":
            result = run_episode_qlearning(agent, env, seed_ep=ep_seed,
                                           eval_mode=True)
        elif agent_name == "sarsa":
            result = run_episode_sarsa(agent, env, seed_ep=ep_seed,
                                       eval_mode=True)
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        rewards.append(result["total_reward"])
        steps.append(result["steps"])
        successes.append(result["success"])
        collisions.append(result["collision"])

    return {
        "rewards":    rewards,
        "steps":      steps,
        "successes":  successes,
        "collisions": collisions,
    }


# ======================================================================= #
#  Đánh giá trên tất cả seeds                                              #
# ======================================================================= #

def evaluate_all_seeds(agent_name: str, cfg: dict,
                       save_dir: str, n_eval_episodes: int) -> dict:
    """
    Tải Q-table (nếu có) cho mỗi seed và đánh giá.

    Returns
    -------
    dict  summary statistics
    """
    seeds   = cfg["evaluation"]["seeds"]
    env_cfg = cfg["env"]
    env     = make_env(cfg)

    all_rewards    = []
    all_steps      = []
    all_successes  = []
    all_collisions = []

    for seed in seeds:
        # Tạo agent
        if agent_name == "random":
            agent = RandomAgent(n_actions=env.n_actions, seed=seed)

        elif agent_name == "heuristic":
            agent = HeuristicAgent(env=env)

        elif agent_name == "q_learning":
            qcfg = cfg["q_learning"]
            agent = QLearningAgent(
                n_states  = env.n_states,
                n_actions = env.n_actions,
                alpha     = qcfg["alpha"],
                gamma     = qcfg["gamma"],
                seed      = seed,
            )
            qtable_path = os.path.join(
                save_dir, f"q_learning_seed{seed}_qtable.npy"
            )
            if os.path.exists(qtable_path):
                agent.load(qtable_path)
                agent.epsilon = 0.0
            else:
                print(f"  [WARN] Q-table không tìm thấy: {qtable_path}")
                print(f"  → Chạy train.py trước!")
                continue

        elif agent_name == "sarsa":
            scfg = cfg["sarsa"]
            agent = SARSAAgent(
                n_states  = env.n_states,
                n_actions = env.n_actions,
                alpha     = scfg["alpha"],
                gamma     = scfg["gamma"],
                seed      = seed,
            )
            qtable_path = os.path.join(
                save_dir, f"sarsa_seed{seed}_qtable.npy"
            )
            if os.path.exists(qtable_path):
                agent.load(qtable_path)
                agent.epsilon = 0.0
            else:
                print(f"  [WARN] Q-table không tìm thấy: {qtable_path}")
                continue

        else:
            raise ValueError(f"Unknown agent: {agent_name}")

        # Đánh giá
        result = evaluate_agent(
            agent, env, agent_name,
            n_episodes=n_eval_episodes,
            seed_offset=seed,
        )

        # Tổng hợp theo từng seed
        all_rewards.append(float(np.mean(result["rewards"])))
        all_steps.append(float(np.mean(result["steps"])))
        all_successes.append(float(np.mean(result["successes"])))
        all_collisions.append(float(np.mean(result["collisions"])))

    if not all_rewards:
        return {}

    summary = {
        "agent": agent_name,
        "n_seeds": len(seeds),
        "n_eval_episodes": n_eval_episodes,
        "reward_mean":     float(np.mean(all_rewards)),
        "reward_std":      float(np.std(all_rewards)),
        "steps_mean":      float(np.mean(all_steps)),
        "steps_std":       float(np.std(all_steps)),
        "success_mean":    float(np.mean(all_successes)),
        "success_std":     float(np.std(all_successes)),
        "collision_mean":  float(np.mean(all_collisions)),
        "collision_std":   float(np.std(all_collisions)),
        "per_seed_rewards":    all_rewards,
        "per_seed_steps":      all_steps,
        "per_seed_successes":  all_successes,
        "per_seed_collisions": all_collisions,
    }
    return summary


# ======================================================================= #
#  In kết quả                                                               #
# ======================================================================= #

def print_summary(summaries: list):
    """In bảng so sánh tất cả agents."""
    print("\n" + "=" * 80)
    print(" KẾT QUẢ ĐÁNH GIÁ – MEAN ± STD (10 SEED)")
    print("=" * 80)
    hdr = f"{'Agent':15s} {'Reward':>16s} {'Steps':>16s} {'Success':>12s} {'Collision':>12s}"
    print(hdr)
    print("-" * 80)
    for s in summaries:
        if not s:
            continue
        print(
            f"{s['agent']:15s} "
            f"{s['reward_mean']:+7.1f}±{s['reward_std']:.1f}  "
            f"{s['steps_mean']:6.1f}±{s['steps_std']:.1f}  "
            f"{s['success_mean']:.2%}±{s['success_std']:.2%}  "
            f"{s['collision_mean']:.2%}±{s['collision_std']:.2%}"
        )
    print("=" * 80)

    # Kiểm tra mục tiêu
    print("\n KIỂM TRA MỤC TIÊU ĐỀ TÀI:")
    for s in summaries:
        if not s or s["agent"] not in ("q_learning", "sarsa"):
            continue
        ok_sr  = "✓" if s["success_mean"]   >= 0.85 else "✗"
        ok_col = "✓" if s["collision_mean"] <= 0.10 else "✗"
        print(f"  {s['agent']:12s}: "
              f"Success {s['success_mean']:.1%} {ok_sr} (target ≥85%)  |  "
              f"Collision {s['collision_mean']:.1%} {ok_col} (target ≤10%)")


# ======================================================================= #
#  CLI                                                                      #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser(description="Đánh giá agents RL")
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic", "q_learning", "sarsa", "all"],
        default="all",
    )
    parser.add_argument("--config", default="experiments/configs.yaml")
    parser.add_argument(
        "--random-map", dest="random_map", action="store_true",
        help="Bật bản đồ vật cản ngẫu nhiên (phải khớp với lúc train)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "extreme"], default=None,
    )
    parser.add_argument("--n-obstacles", dest="n_obstacles", type=int, default=None)
    parser.add_argument("--map-seed", dest="map_seed", type=int, default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Ghi đè cấu hình bản đồ từ CLI (cần khớp cấu hình lúc train)
    if args.random_map:
        cfg["env"]["random_map"] = True
    if args.difficulty is not None:
        cfg["env"]["difficulty"] = args.difficulty
    if args.n_obstacles is not None:
        cfg["env"]["n_obstacles"] = args.n_obstacles
    if args.map_seed is not None:
        cfg["env"]["map_seed"] = args.map_seed

    save_dir = os.path.join(root, cfg.get("logging", {}).get("save_dir",
                                                              "experiments/results"))
    n_eval   = cfg["evaluation"]["n_eval_episodes"]

    agent_list = (
        ["random", "heuristic", "q_learning", "sarsa"]
        if args.agent == "all"
        else [args.agent]
    )

    print("=" * 60)
    print(" ĐÁNH GIÁ – EPSILON=0, 10 SEED")
    print("=" * 60)

    summaries = []
    for name in agent_list:
        print(f"\n>>> Đánh giá: {name}")
        s = evaluate_all_seeds(name, cfg, save_dir, n_eval)
        summaries.append(s)

        # Lưu summary
        if s:
            out = os.path.join(save_dir, f"eval_summary_{name}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)

    print_summary(summaries)

    # Lưu tổng hợp
    all_out = os.path.join(save_dir, "eval_summary_all.json")
    with open(all_out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\n Kết quả đã lưu tại: {all_out}")


if __name__ == "__main__":
    main()
