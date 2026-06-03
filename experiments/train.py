"""
train.py
--------
Script huấn luyện tất cả agents (Random, Heuristic, Q-Learning, SARSA)
trên môi trường DirectionalCarEnv.

Chạy:
    python experiments/train.py                   # dùng configs.yaml mặc định
    python experiments/train.py --agent q_learning --seed 42

Kết quả lưu tại: experiments/results/
"""

import argparse
import json
import os
import sys
import time

# Fix Unicode encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import yaml

# Thêm project root vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import DirectionalCarEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent


# ======================================================================= #
#  Hàm train một agent trong một lần chạy (một seed)                       #
# ======================================================================= #

def run_episode_random_or_heuristic(agent, env, seed_ep=None):
    """Chạy một episode cho RandomAgent hoặc HeuristicAgent."""
    obs, _ = env.reset(seed=seed_ep)
    total_reward = 0.0
    done = False
    collision = False
    reached = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if info.get("collision"):
            collision = True
        if info.get("reached_goal"):
            reached = True

    return {
        "total_reward": total_reward,
        "steps": env.steps,
        "success": int(reached),
        "collision": int(collision),
    }


def run_episode_qlearning(agent: QLearningAgent, env: DirectionalCarEnv,
                          seed_ep=None, eval_mode=False):
    """Chạy một episode Q-Learning (off-policy)."""
    obs, _ = env.reset(seed=seed_ep)
    total_reward = 0.0
    done = False
    collision = False
    reached = False

    while not done:
        action = agent.select_action(obs, eval_mode=eval_mode)
        next_obs, reward, terminated, truncated, info = env.step(action)

        if not eval_mode:
            agent.update(obs, action, reward, next_obs, terminated)

        total_reward += reward
        obs = next_obs
        done = terminated or truncated

        if info.get("collision"):
            collision = True
        if info.get("reached_goal"):
            reached = True

    return {
        "total_reward": total_reward,
        "steps": env.steps,
        "success": int(reached),
        "collision": int(collision),
    }


def run_episode_sarsa(agent: SARSAAgent, env: DirectionalCarEnv,
                      seed_ep=None, eval_mode=False):
    """
    Chạy một episode SARSA (on-policy).
    SARSA yêu cầu chọn a' TRƯỚC KHI cập nhật Q(s,a).
    """
    obs, _ = env.reset(seed=seed_ep)
    action = agent.select_action(obs, eval_mode=eval_mode)
    total_reward = 0.0
    done = False
    collision = False
    reached = False

    while not done:
        next_obs, reward, terminated, truncated, info = env.step(action)

        if not eval_mode:
            next_action = agent.select_action(next_obs, eval_mode=False) \
                if not (terminated or truncated) else 0
            agent.update(obs, action, reward, next_obs, terminated,
                         next_action=next_action)
        else:
            next_action = agent.select_action(next_obs, eval_mode=True)

        total_reward += reward
        obs = next_obs
        action = next_action
        done = terminated or truncated

        if info.get("collision"):
            collision = True
        if info.get("reached_goal"):
            reached = True

    return {
        "total_reward": total_reward,
        "steps": env.steps,
        "success": int(reached),
        "collision": int(collision),
    }


# ======================================================================= #
#  Hàm train chính                                                          #
# ======================================================================= #

def train(
    agent_name: str,
    cfg: dict,
    seed: int,
    save_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Huấn luyện một agent trong một seed.

    Parameters
    ----------
    agent_name : str     "random" | "heuristic" | "q_learning" | "sarsa"
    cfg        : dict    Nội dung configs.yaml
    seed       : int     Seed cho môi trường và agent
    save_dir   : str     Thư mục lưu Q-table và metrics
    verbose    : bool    In progress ra terminal

    Returns
    -------
    dict  Metrics toàn bộ quá trình huấn luyện
    """
    env_cfg = cfg["env"]
    env = DirectionalCarEnv(max_steps=env_cfg["max_steps"])

    n_episodes = 1000  # cho random/heuristic (không có epsilon decay)

    # ---- Khởi tạo agent ----
    if agent_name == "random":
        agent = RandomAgent(n_actions=env.n_actions, seed=seed)
        run_ep = run_episode_random_or_heuristic

    elif agent_name == "heuristic":
        agent = HeuristicAgent(env=env)
        run_ep = run_episode_random_or_heuristic

    elif agent_name == "q_learning":
        qcfg = cfg["q_learning"]
        n_episodes = qcfg["n_episodes"]
        agent = QLearningAgent(
            n_states      = env.n_states,
            n_actions     = env.n_actions,
            alpha         = qcfg["alpha"],
            gamma         = qcfg["gamma"],
            epsilon_start = qcfg["epsilon_start"],
            epsilon_end   = qcfg["epsilon_end"],
            epsilon_decay = qcfg["epsilon_decay"],
            seed          = seed,
        )
        run_ep = run_episode_qlearning

    elif agent_name == "sarsa":
        scfg = cfg["sarsa"]
        n_episodes = scfg["n_episodes"]
        agent = SARSAAgent(
            n_states      = env.n_states,
            n_actions     = env.n_actions,
            alpha         = scfg["alpha"],
            gamma         = scfg["gamma"],
            epsilon_start = scfg["epsilon_start"],
            epsilon_end   = scfg["epsilon_end"],
            epsilon_decay = scfg["epsilon_decay"],
            seed          = seed,
        )
        run_ep = run_episode_sarsa

    else:
        raise ValueError(f"Agent không hợp lệ: {agent_name}")

    # ---- Huấn luyện ----
    metrics = {
        "episode_rewards":    [],
        "episode_steps":      [],
        "episode_successes":  [],
        "episode_collisions": [],
        "epsilon_history":    [],
        "rolling_success_rate": [],
    }

    window     = 100
    log_interval = cfg.get("logging", {}).get("log_interval", 100)
    t_start    = time.time()

    for ep in range(n_episodes):
        result = run_ep(agent, env)

        metrics["episode_rewards"].append(result["total_reward"])
        metrics["episode_steps"].append(result["steps"])
        metrics["episode_successes"].append(result["success"])
        metrics["episode_collisions"].append(result["collision"])

        epsilon = getattr(agent, "epsilon", 0.0)
        metrics["epsilon_history"].append(epsilon)

        # Rolling success rate (cửa sổ 100 episode)
        if ep >= window - 1:
            sr = float(np.mean(metrics["episode_successes"][ep - window + 1 : ep + 1]))
        else:
            sr = float(np.mean(metrics["episode_successes"][: ep + 1]))
        metrics["rolling_success_rate"].append(sr)

        agent.decay_epsilon()

        if verbose and (ep + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            print(
                f"[{agent_name:12s}][seed={seed}] ep={ep+1:5d}/{n_episodes} | "
                f"ε={epsilon:.4f} | "
                f"rew={np.mean(metrics['episode_rewards'][-100:]):+7.1f} | "
                f"sr={sr:.2%} | "
                f"steps={np.mean(metrics['episode_steps'][-100:]):.1f} | "
                f"t={elapsed:.1f}s"
            )

    # ---- Lưu kết quả ----
    os.makedirs(save_dir, exist_ok=True)

    # Lưu Q-table (chỉ RL agents)
    if agent_name in ("q_learning", "sarsa"):
        qtable_path = os.path.join(save_dir, f"{agent_name}_seed{seed}_qtable.npy")
        agent.save(qtable_path)
        if verbose:
            print(f"  → Q-table lưu tại: {qtable_path}")

    # Lưu metrics
    metrics_path = os.path.join(save_dir, f"metrics_{agent_name}_seed{seed}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    if verbose:
        print(f"  → Metrics lưu tại: {metrics_path}")

    return metrics


# ======================================================================= #
#  CLI                                                                      #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện agents RL")
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic", "q_learning", "sarsa", "all"],
        default="all",
        help="Agent cần huấn luyện (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed cụ thể (default: chạy tất cả seed từ configs)",
    )
    parser.add_argument(
        "--config",
        default="experiments/configs.yaml",
        help="Đường dẫn tới file config",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Tắt verbose logging",
    )
    args = parser.parse_args()

    # Tải config
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        cfg.get("logging", {}).get("save_dir", "experiments/results")
    )

    seeds = [args.seed] if args.seed is not None else cfg["evaluation"]["seeds"]
    agents = (
        ["random", "heuristic", "q_learning", "sarsa"]
        if args.agent == "all"
        else [args.agent]
    )

    print("=" * 60)
    print(" HUẤN LUYỆN AGENTS – ĐỀ TÀI 15: XE TỰ HÀNH MINI")
    print(f" Agents : {agents}")
    print(f" Seeds  : {seeds}")
    print("=" * 60)

    for agent_name in agents:
        for seed in seeds:
            print(f"\n>>> Bắt đầu: {agent_name} | seed={seed}")
            train(
                agent_name=agent_name,
                cfg=cfg,
                seed=seed,
                save_dir=save_dir,
                verbose=not args.quiet,
            )

    print("\n" + "=" * 60)
    print(" HOÀN THÀNH HUẤN LUYỆN")
    print("=" * 60)


if __name__ == "__main__":
    main()
