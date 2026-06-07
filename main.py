"""
main.py
-------
Entry point chính của dự án Đề tài 15: Xe tự hành mini trong grid có hướng.

Sử dụng:
    python main.py                        # Hiển thị menu
    python main.py train                  # Huấn luyện tất cả agents
    python main.py train --agent q_learning --seed 0
    python main.py evaluate               # Đánh giá sau khi train
    python main.py demo                   # Mở dashboard tương tác
    python main.py demo --train_first     # Train nhanh rồi mở dashboard
    python main.py test                   # Chạy unit tests
    python main.py plots                  # Vẽ tất cả hình
    python main.py quickrun               # Demo nhanh terminal (không GUI)
"""

import argparse
import os
import subprocess
import sys

# Thêm project root vào sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ======================================================================= #
#  Quick terminal demo (không cần matplotlib GUI)                          #
# ======================================================================= #

def quickrun(n_steps: int = 30, seed: int = 42):
    """Chạy Q-Learning agent vài bước và in ra terminal."""
    import yaml
    from envs.custom_env import DirectionalCarEnv
    from agents.q_learning import QLearningAgent
    from agents.heuristic_agent import HeuristicAgent
    from agents.random_agent import RandomAgent

    print("=" * 50)
    print(" QUICKRUN – Demo terminal (không GUI)")
    print("=" * 50)

    env = DirectionalCarEnv(max_steps=50)

    agents_to_demo = [
        ("Random",    RandomAgent(n_actions=env.n_actions, seed=seed)),
        ("Heuristic", HeuristicAgent(env=env)),
    ]

    # Load Q-Learning nếu đã train
    save_dir = os.path.join(ROOT, "experiments", "results")
    ql = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, seed=seed)
    qtable_path = os.path.join(save_dir, f"q_learning_seed0_qtable.npy")
    if os.path.exists(qtable_path):
        ql.load(qtable_path)
        ql.epsilon = 0.0
        agents_to_demo.append(("Q-Learning", ql))
        print(f"  Loaded Q-table: {qtable_path}\n")
    else:
        print("  [INFO] Q-table chưa có, chạy 'python main.py train' trước.\n")

    for agent_name, agent in agents_to_demo:
        print(f"\n{'─'*40}")
        print(f" Agent: {agent_name}")
        print(f"{'─'*40}")

        obs, _ = env.reset(seed=seed)
        print(env.render())

        total_reward = 0.0
        for step in range(n_steps):
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            action_name = env.ACTION_NAMES[action]
            print(f"\n Hành động: {action_name:10s} | Reward: {reward:+.0f} | "
                  f"Total: {total_reward:+.1f}")
            print(env.render())

            if terminated or truncated:
                if info.get("reached_goal"):
                    print(f"\n ✅ ĐẾN ĐÍCH! Tổng reward: {total_reward:+.1f}")
                elif info.get("collision"):
                    print(f"\n ❌ VA CHẠM! Tổng reward: {total_reward:+.1f}")
                else:
                    print(f"\n ⏰ Hết bước! Tổng reward: {total_reward:+.1f}")
                break
        else:
            print(f"\n Hết {n_steps} bước demo. Tổng reward: {total_reward:+.1f}")


# ======================================================================= #
#  Menu tương tác                                                          #
# ======================================================================= #

def print_menu():
    print("""
╔══════════════════════════════════════════════════════╗
║   ĐỀ TÀI 15 – Xe tự hành mini trong grid có hướng   ║
║   Môi trường: 7×7 grid | State: 588 | Action: 4      ║
╠══════════════════════════════════════════════════════╣
║  Lệnh:                                               ║
║    train      – Huấn luyện tất cả agents (10 seed)   ║
║    evaluate   – Đánh giá và báo cáo mean±std         ║
║    demo       – Mở dashboard tương tác (matplotlib)  ║
║    demo3d     – Mở bản đồ 3D xoay được (matplotlib)  ║
║    test       – Chạy unit tests                      ║
║    plots      – Vẽ learning curves và so sánh        ║
║    quickrun   – Demo nhanh trên terminal             ║
╚══════════════════════════════════════════════════════╝
""")


# ======================================================================= #
#  CLI                                                                      #
# ======================================================================= #

def main():
    parser = argparse.ArgumentParser(
        description="Đề tài 15 – Xe tự hành mini trong grid có hướng",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["train", "evaluate", "demo", "demo3d", "test", "plots", "quickrun"],
        default=None,
        help="Lệnh cần thực hiện",
    )
    # Forward extra args to subcommands
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Tham số thêm cho subcommand")

    args = parser.parse_args()

    if args.command is None:
        print_menu()
        cmd = input("Nhập lệnh (hoặc Enter để thoát): ").strip().lower()
        if not cmd:
            return
        args.command = cmd

    if args.command == "train":
        extra = args.extra_args if args.extra_args else []
        script = os.path.join(ROOT, "experiments", "train.py")
        subprocess.run([sys.executable, script] + extra)

    elif args.command == "evaluate":
        extra = args.extra_args if args.extra_args else []
        script = os.path.join(ROOT, "experiments", "evaluate.py")
        subprocess.run([sys.executable, script] + extra)

    elif args.command == "demo":
        extra = args.extra_args if args.extra_args else []
        script = os.path.join(ROOT, "dashboard", "app.py")
        subprocess.run([sys.executable, script] + extra)

    elif args.command == "demo3d":
        extra = args.extra_args if args.extra_args else []
        script = os.path.join(ROOT, "main3d.py")
        subprocess.run([sys.executable, script] + extra)

    elif args.command == "test":
        print(" Chạy unit tests...\n")
        ret = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=ROOT,
        )
        sys.exit(ret.returncode)

    elif args.command == "plots":
        import yaml
        from visualization.plots import plot_all
        cfg_path = os.path.join(ROOT, "experiments", "configs.yaml")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        save_dir    = os.path.join(ROOT, cfg["logging"]["save_dir"])
        figures_dir = os.path.join(ROOT, cfg["logging"]["plot_dir"])
        seeds       = cfg["evaluation"]["seeds"]
        plot_all(save_dir, seeds, figures_dir)

    elif args.command == "quickrun":
        quickrun()

    else:
        print(f"Lệnh không hợp lệ: {args.command}")
        print_menu()


if __name__ == "__main__":
    main()
