"""
dashboard/app.py  –  Giao diện demo tương tác
================================================
Đề tài 15: Xe tự hành mini trong grid 7×7 có hướng

Layout:
  ┌──────────────────────────────────────────────────────┐
  │ TIÊU ĐỀ                                 [✕]          │
  ├──────────────┬──────────────────────────┬────────────┤
  │              │     GRID 7×7             │  INFO      │
  │  CONTROLS    │   (matplotlib canvas)    │  PANEL     │
  │  - Agent     │                          │            │
  │  - Goal      ├──────────────────────────┤            │
  │  - Buttons   │  [Tab: Curves | Policy | Comparison]  │
  └──────────────┴───────────────────────────────────────┘

Chạy:
    python dashboard/app.py
    python dashboard/app.py --train_first   (train 3000 ep trước)
"""

import argparse
import json
import os
import sys
import tkinter as tk
from tkinter import ttk, font as tkfont

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import DirectionalCarEnv
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent

# ── Màu sắc giao diện ──────────────────────────────────────────────────
THEME = {
    "bg":           "#1E1E2E",   # nền chính (dark)
    "panel":        "#2A2A3E",   # nền panel
    "accent":       "#7C3AED",   # tím accent
    "accent2":      "#06B6D4",   # cyan accent
    "success":      "#10B981",   # xanh lá
    "danger":       "#EF4444",   # đỏ
    "warning":      "#F59E0B",   # cam
    "text":         "#E2E8F0",   # chữ sáng
    "text_dim":     "#94A3B8",   # chữ mờ
    "border":       "#3F3F5A",   # viền
}

# ── Màu grid ──────────────────────────────────────────────────────────
CELL_EMPTY    = "#FFFFFF"
CELL_OBS      = "#374151"
CELL_GOAL_ON  = "#10B981"
CELL_GOAL_OFF = "#D1FAE5"
CELL_TRAIL    = "#BAE6FD"
CELL_CAR      = "#EF4444"

AGENT_COLORS  = {
    "Random":    "#94A3B8",
    "Heuristic": "#F59E0B",
    "Q-Learning":"#3B82F6",
    "SARSA":     "#10B981",
}


# ======================================================================= #
#  Quick-train helper                                                       #
# ======================================================================= #

def load_train_config() -> dict:
    """
    Đọc siêu tham số huấn luyện từ experiments/configs.yaml.

    Trả về dict config (rỗng nếu đọc lỗi → caller dùng giá trị mặc định).
    Nhờ vậy dashboard train cùng tham số với train chính (experiments/train.py),
    chỉnh configs.yaml là cả hai nơi đổi theo, không cần sửa code.
    """
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "experiments", "configs.yaml")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Không đọc được configs.yaml ({e}); dùng tham số mặc định.")
        return {}


def _hp(cfg: dict, agent_key: str) -> dict:
    """Lấy nhóm siêu tham số của một agent, kèm giá trị mặc định an toàn."""
    hp = (cfg or {}).get(agent_key, {}) or {}
    return {
        "alpha":         hp.get("alpha", 0.1),
        "gamma":         hp.get("gamma", 0.95),
        "epsilon_start": hp.get("epsilon_start", 1.0),
        "epsilon_end":   hp.get("epsilon_end", 0.01),
        "epsilon_decay": hp.get("epsilon_decay", 0.997),
        "n_episodes":    hp.get("n_episodes", 3000),
    }


def quick_train(n_ep: int = None, cfg: dict = None) -> dict:
    """Train nhanh Q-Learning & SARSA dùng siêu tham số từ configs.yaml."""
    from experiments.train import run_episode_qlearning, run_episode_sarsa
    cfg = cfg if cfg is not None else load_train_config()
    env = DirectionalCarEnv()
    result = {}
    for Cls, fn, key, cfg_key in [
        (QLearningAgent, run_episode_qlearning, "Q-Learning", "q_learning"),
        (SARSAAgent,     run_episode_sarsa,     "SARSA",      "sarsa"),
    ]:
        hp = _hp(cfg, cfg_key)
        episodes = n_ep if n_ep is not None else hp["n_episodes"]
        a = Cls(env.n_states, env.n_actions,
                alpha=hp["alpha"], gamma=hp["gamma"],
                epsilon_start=hp["epsilon_start"], epsilon_end=hp["epsilon_end"],
                epsilon_decay=hp["epsilon_decay"], seed=0)
        for _ in range(episodes):
            fn(a, env)
            a.decay_epsilon()
        a.epsilon = 0.0
        result[key] = a
    return result


# ======================================================================= #
#  Main App                                                                 #
# ======================================================================= #

class App(tk.Tk):
    """Cửa sổ chính của dashboard."""

    AGENTS = ["Random", "Heuristic", "Q-Learning", "SARSA"]
    GOALS  = ["G0 – (0, 6)", "G1 – (3, 6)", "G2 – (6, 0)"]

    def __init__(self, pretrained: dict = None):
        super().__init__()

        # ── Cài đặt cửa sổ ──
        self.title("Đề tài 15 – Xe tự hành mini trong grid 7×7")
        self.configure(bg=THEME["bg"])
        self.resizable(True, True)
        self.minsize(1100, 680)
        self._center_window(1180, 720)

        # ── Môi trường & agents ──
        self._cfg = load_train_config()
        self.env = DirectionalCarEnv(max_steps=200)
        self._agents = self._init_agents(pretrained)
        self._sel_agent  = tk.StringVar(value="Q-Learning")
        self._sel_goal   = tk.IntVar(value=0)
        self._auto_id    = None        # after() callback id
        self._auto_running = False     # cờ vòng lặp auto-run
        self._auto_speed = tk.IntVar(value=150)   # ms

        # ── Trạng thái episode ──
        self._obs        = None
        self._trail: list = []
        self._ep_reward  = 0.0
        self._last_action = None
        self._last_reward = 0.0
        self._done        = False
        self._ep_count    = 0

        # ── Xây dựng giao diện ──
        self._build_ui()
        self._reset_episode()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    #  Khởi tạo agents                                                     #
    # ------------------------------------------------------------------ #

    def _init_agents(self, pretrained):
        env = self.env

        # Khởi tạo từng agent an toàn: agent nào chưa cài (NotImplementedError)
        # hoặc lỗi khi tạo sẽ để None → giao diện vẫn mở được bình thường.
        def _safe(factory):
            try:
                return factory()
            except Exception as e:
                print(f"[INFO] Agent chưa sẵn sàng: {e.__class__.__name__}")
                return None

        agents = {
            "Random":    _safe(lambda: RandomAgent(env.n_actions, seed=42)),
            "Heuristic": _safe(lambda: HeuristicAgent(env)),
            "Q-Learning": _safe(lambda: QLearningAgent(env.n_states, env.n_actions,
                                                       alpha=0.1, gamma=0.95, seed=42)),
            "SARSA":      _safe(lambda: SARSAAgent(env.n_states, env.n_actions,
                                                   alpha=0.1, gamma=0.95, seed=42)),
        }
        # Load pre-trained (chỉ khi agent đã được cài đặt)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(root, "experiments", "results")
        for key, fname in [("Q-Learning", "q_learning"), ("SARSA", "sarsa")]:
            if agents[key] is None:
                continue
            p = os.path.join(save_dir, f"{fname}_seed0_qtable.npy")
            if os.path.exists(p):
                try:
                    agents[key].load(p)
                    agents[key].epsilon = 0.0
                except Exception:
                    pass
        # Override with pretrained
        if pretrained:
            for key, ag in pretrained.items():
                if agents.get(key) is None:
                    continue
                try:
                    agents[key].Q = ag.Q.copy()
                    agents[key].epsilon = 0.0
                except Exception:
                    pass
        return agents

    # ------------------------------------------------------------------ #
    #  Build UI                                                            #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=THEME["accent"], height=42)
        hdr.pack(fill="x", side="top")
        tk.Label(hdr,
                 text="🚗  Xe Tự Hành Mini  –  Grid 7×7  –  Đề tài 15",
                 bg=THEME["accent"], fg="white",
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=16, pady=8)
        tk.Label(hdr, text="RL: Q-Learning & SARSA",
                 bg=THEME["accent"], fg="#C4B5FD",
                 font=("Segoe UI", 10)).pack(side="right", padx=16)

        # ── Nội dung chính ──
        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Cột trái: controls
        left = tk.Frame(body, bg=THEME["panel"], width=210,
                        relief="flat", bd=0)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 6))
        left.pack_propagate(False)
        self._build_controls(left)

        # Cột giữa + phải
        mid = tk.Frame(body, bg=THEME["bg"])
        mid.grid(row=0, column=1, sticky="nsew")
        mid.rowconfigure(0, weight=3)
        mid.rowconfigure(1, weight=2)
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=1)

        # Grid canvas (top-left)
        grid_frame = tk.Frame(mid, bg=THEME["panel"],
                              relief="flat", highlightthickness=1,
                              highlightbackground=THEME["border"])
        grid_frame.grid(row=0, column=0, sticky="nsew",
                        padx=(0, 6), pady=(0, 6))
        self._build_grid_canvas(grid_frame)

        # Info panel (top-right)
        info_frame = tk.Frame(mid, bg=THEME["panel"], width=200,
                              relief="flat", highlightthickness=1,
                              highlightbackground=THEME["border"])
        info_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 6))
        info_frame.pack_propagate(False)
        self._build_info_panel(info_frame)

        # Tab panel (bottom)
        tab_frame = tk.Frame(mid, bg=THEME["panel"],
                             relief="flat", highlightthickness=1,
                             highlightbackground=THEME["border"])
        tab_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self._build_tab_panel(tab_frame)

        # ── Status bar ──
        self._status_var = tk.StringVar(value="Sẵn sàng")
        sb = tk.Label(self, textvariable=self._status_var,
                      bg=THEME["border"], fg=THEME["text_dim"],
                      font=("Consolas", 9), anchor="w", padx=10)
        sb.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------ #
    #  Controls (cột trái)                                                 #
    # ------------------------------------------------------------------ #

    def _build_controls(self, parent):
        pad = {"padx": 12, "pady": 6}

        tk.Label(parent, text="⚙ ĐIỀU KHIỂN",
                 bg=THEME["panel"], fg=THEME["accent2"],
                 font=("Segoe UI", 10, "bold")).pack(**pad, anchor="w")

        self._sep(parent)

        # Agent selector
        tk.Label(parent, text="Chọn Agent:",
                 bg=THEME["panel"], fg=THEME["text"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(8,2), anchor="w")

        for name in self.AGENTS:
            color = AGENT_COLORS[name]
            rb = tk.Radiobutton(
                parent, text=name,
                variable=self._sel_agent, value=name,
                command=self._on_agent_change,
                bg=THEME["panel"], fg=THEME["text"],
                selectcolor=THEME["bg"],
                activebackground=THEME["panel"],
                activeforeground=color,
                indicatoron=0,
                relief="flat", bd=0,
                font=("Segoe UI", 9),
                cursor="hand2",
                padx=10, pady=4,
                width=16, anchor="w",
            )
            rb.pack(padx=12, pady=1, fill="x")
            rb.bind("<Enter>", lambda e, r=rb, c=color: r.config(fg=c))
            rb.bind("<Leave>", lambda e, r=rb: r.config(fg=THEME["text"]))

        self._sep(parent)

        # Goal selector
        tk.Label(parent, text="Chọn Goal:",
                 bg=THEME["panel"], fg=THEME["text"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(8,2), anchor="w")

        for i, glabel in enumerate(self.GOALS):
            rb = tk.Radiobutton(
                parent, text=glabel,
                variable=self._sel_goal, value=i,
                command=self._on_goal_change,
                bg=THEME["panel"], fg=THEME["text"],
                selectcolor=THEME["bg"],
                activebackground=THEME["panel"],
                indicatoron=0, relief="flat", bd=0,
                font=("Segoe UI", 9),
                cursor="hand2",
                padx=10, pady=3, width=16, anchor="w",
            )
        self._sep(parent)

        # Map / difficulty selector
        tk.Label(parent, text="Bản đồ / Độ khó:",
                 bg=THEME["panel"], fg=THEME["text"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(8, 2), anchor="w")

        self._sel_difficulty = tk.StringVar(value="Cố định")
        diff_options = ["Cố định", "easy", "medium", "hard", "extreme"]
        om = tk.OptionMenu(parent, self._sel_difficulty, *diff_options)
        om.config(bg=THEME["bg"], fg=THEME["text"], font=("Segoe UI", 9),
                  activebackground=THEME["accent"], activeforeground="white",
                  highlightthickness=0, relief="flat", cursor="hand2", width=14)
        om["menu"].config(bg=THEME["panel"], fg=THEME["text"])
        om.pack(padx=12, pady=2, fill="x")

        self._btn_newmap = tk.Button(
            parent, text="🗺  Tạo bản đồ & train lại",
            bg=THEME["accent"], fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
            cursor="hand2", pady=6,
            command=self._on_change_map)
        self._btn_newmap.pack(padx=12, pady=4, fill="x")

        self._sep(parent)

        # Speed slider
        tk.Label(parent, text="Tốc độ Auto (ms):",
                 bg=THEME["panel"], fg=THEME["text"],
                 font=("Segoe UI", 9)).pack(padx=12, pady=(8,0), anchor="w")
        sl = tk.Scale(
            parent, from_=50, to=800,
            variable=self._auto_speed,
            orient="horizontal", bg=THEME["panel"],
            fg=THEME["text"], troughcolor=THEME["bg"],
            highlightthickness=0, bd=0,
            font=("Segoe UI", 8),
        )
        sl.pack(padx=12, fill="x")

        self._sep(parent)

        # Buttons
        btn_cfg = dict(
            font=("Segoe UI", 10, "bold"),
            relief="flat", bd=0,
            cursor="hand2", pady=8,
        )

        # Nút CHẠY chính – chạy thuật toán đang chọn
        self._btn_auto = tk.Button(
            parent, text="▶  CHẠY thuật toán",
            bg=THEME["success"], fg="white",
            command=self._on_auto, **btn_cfg)
        self._btn_auto.pack(padx=12, pady=4, fill="x")

        self._btn_step = tk.Button(
            parent, text="⏭  Bước tiếp",
            bg=THEME["accent2"], fg="white",
            command=self._on_step, **btn_cfg)
        self._btn_step.pack(padx=12, pady=4, fill="x")

        self._btn_stop_auto = tk.Button(
            parent, text="⏹  Dừng",
            bg=THEME["border"], fg=THEME["text"],
            command=self._stop_auto, **btn_cfg)
        self._btn_stop_auto.pack(padx=12, pady=4, fill="x")

        self._btn_reset = tk.Button(
            parent, text="↺  Episode mới",
            bg="#374151", fg=THEME["text"],
            command=self._on_reset, **btn_cfg)
        self._btn_reset.pack(padx=12, pady=4, fill="x")

        self._sep(parent)

        # Thống kê nhanh
        tk.Label(parent, text="📈 THỐNG KÊ NHANH",
                 bg=THEME["panel"], fg=THEME["accent2"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(8,2), anchor="w")

        self._stat_ep   = tk.StringVar(value="Episode: 0")
        self._stat_win  = tk.StringVar(value="Thành công: 0")
        self._stat_rate = tk.StringVar(value="Tỷ lệ: –")

        for var in (self._stat_ep, self._stat_win, self._stat_rate):
            tk.Label(parent, textvariable=var,
                     bg=THEME["panel"], fg=THEME["text_dim"],
                     font=("Consolas", 9)).pack(padx=12, anchor="w")

        self._wins = 0

    def _sep(self, parent):
        tk.Frame(parent, bg=THEME["border"], height=1).pack(
            fill="x", padx=8, pady=4)

    # ------------------------------------------------------------------ #
    #  Grid Canvas                                                         #
    # ------------------------------------------------------------------ #

    def _build_grid_canvas(self, parent):
        self._fig_grid = Figure(figsize=(4.6, 4.6), facecolor=THEME["panel"])
        self._ax_grid  = self._fig_grid.add_subplot(111)
        self._ax_grid.set_facecolor(THEME["panel"])
        self._fig_grid.tight_layout(pad=0.5)

        canvas = FigureCanvasTkAgg(self._fig_grid, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        canvas.get_tk_widget().configure(bg=THEME["panel"])
        self._canvas_grid = canvas

    # ------------------------------------------------------------------ #
    #  Info Panel                                                          #
    # ------------------------------------------------------------------ #

    def _build_info_panel(self, parent):
        tk.Label(parent, text="ℹ  THÔNG TIN",
                 bg=THEME["panel"], fg=THEME["accent2"],
                 font=("Segoe UI", 10, "bold")).pack(padx=10, pady=(10, 4), anchor="w")

        tk.Frame(parent, bg=THEME["border"], height=1).pack(fill="x", padx=8, pady=2)

        self._info_vars = {}
        fields = [
            ("pos",     "Vị trí"),
            ("heading", "Hướng"),
            ("goal",    "Goal"),
            ("dist",    "Khoảng cách"),
            ("",        ""),
            ("action",  "Hành động"),
            ("reward",  "Reward"),
            ("total",   "Tổng reward"),
            ("steps",   "Bước"),
            ("",        ""),
            ("status",  "Trạng thái"),
        ]

        for key, label in fields:
            if not key:
                tk.Frame(parent, bg=THEME["border"], height=1).pack(
                    fill="x", padx=8, pady=3)
                continue
            row = tk.Frame(parent, bg=THEME["panel"])
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=f"{label}:", width=12, anchor="w",
                     bg=THEME["panel"], fg=THEME["text_dim"],
                     font=("Segoe UI", 8)).pack(side="left")
            var = tk.StringVar(value="–")
            tk.Label(row, textvariable=var, anchor="w",
                     bg=THEME["panel"], fg=THEME["text"],
                     font=("Consolas", 9)).pack(side="left")
            self._info_vars[key] = var

        # Reward bar
        tk.Frame(parent, bg=THEME["border"], height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(parent, text="Reward episode:",
                 bg=THEME["panel"], fg=THEME["text_dim"],
                 font=("Segoe UI", 8)).pack(padx=10, anchor="w")
        self._bar_canvas = tk.Canvas(parent, height=16, bg=THEME["bg"],
                                     highlightthickness=0)
        self._bar_canvas.pack(fill="x", padx=10, pady=(2, 8))

    # ------------------------------------------------------------------ #
    #  Tab Panel (bottom)                                                  #
    # ------------------------------------------------------------------ #

    def _build_tab_panel(self, parent):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background=THEME["panel"],
                        borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=THEME["border"],
                        foreground=THEME["text_dim"],
                        padding=[12, 4],
                        font=("Segoe UI", 9))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", THEME["accent"])],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(parent, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 1: Learning Curves
        tab_lc = tk.Frame(nb, bg=THEME["panel"])
        nb.add(tab_lc, text="📈 Learning Curves")
        self._build_lc_tab(tab_lc)

        # Tab 2: Policy
        tab_pol = tk.Frame(nb, bg=THEME["panel"])
        nb.add(tab_pol, text="🗺 Policy")
        self._build_policy_tab(tab_pol)

        # Tab 3: Comparison
        tab_cmp = tk.Frame(nb, bg=THEME["panel"])
        nb.add(tab_cmp, text="📊 So sánh")
        self._build_comparison_tab(tab_cmp)

        # Tab 4: Replay trail
        tab_trail = tk.Frame(nb, bg=THEME["panel"])
        nb.add(tab_trail, text="🔄 Replay")
        self._build_replay_tab(tab_trail)

        self._nb = nb

    # ── Tab: Learning Curves ─────────────────────────────────────────

    def _build_lc_tab(self, parent):
        self._fig_lc = Figure(figsize=(9, 2.4), facecolor=THEME["panel"])
        self._ax_lc1 = self._fig_lc.add_subplot(121)
        self._ax_lc2 = self._fig_lc.add_subplot(122)
        self._fig_lc.tight_layout(pad=1.2)

        canvas = FigureCanvasTkAgg(self._fig_lc, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas_lc = canvas
        self._draw_lc()

    def _draw_lc(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(root, "experiments", "results")

        for ax, metric, label in [
            (self._ax_lc1, "episode_rewards",   "Reward"),
            (self._ax_lc2, "episode_successes",  "Success Rate"),
        ]:
            ax.clear()
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors=THEME["text_dim"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(THEME["border"])

            found = False
            for name, color in AGENT_COLORS.items():
                key = name.lower().replace("-", "_").replace(" ", "_")
                path = os.path.join(save_dir, f"metrics_{key}_seed0.json")
                if not os.path.exists(path):
                    continue
                with open(path, encoding="utf-8") as f:
                    m = json.load(f)
                data = np.array(m[metric], dtype=float)
                n = len(data)
                window = 100
                rolled = np.full(n, np.nan)
                for i in range(n):
                    s = max(0, i - window + 1)
                    rolled[i] = np.mean(data[s:i+1])
                ax.plot(rolled, color=color, label=name, linewidth=1.5, alpha=0.9)
                found = True

            if metric == "episode_successes":
                ax.axhline(0.85, color="#EF4444", linewidth=1,
                           linestyle="--", alpha=0.7, label="Target 85%")
                ax.set_ylim(0, 1.05)
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

            ax.set_title(label, color=THEME["text"], fontsize=9, pad=4)
            ax.set_xlabel("Episode", color=THEME["text_dim"], fontsize=7)
            ax.grid(True, alpha=0.15, color=THEME["border"])
            if found:
                ax.legend(fontsize=7, loc="lower right",
                          facecolor=THEME["panel"],
                          labelcolor=THEME["text"], edgecolor=THEME["border"])
            else:
                ax.text(0.5, 0.5, "Chạy  python main.py train  trước",
                        transform=ax.transAxes, ha="center", va="center",
                        color=THEME["text_dim"], fontsize=8)

        self._fig_lc.tight_layout(pad=1.2)
        self._canvas_lc.draw()

    # ── Tab: Policy ──────────────────────────────────────────────────

    def _build_policy_tab(self, parent):
        ctrl = tk.Frame(parent, bg=THEME["panel"])
        ctrl.pack(fill="x", padx=8, pady=4)

        tk.Label(ctrl, text="Heading:", bg=THEME["panel"],
                 fg=THEME["text_dim"], font=("Segoe UI", 9)).pack(side="left", padx=4)
        self._pol_heading = tk.IntVar(value=0)
        heading_names = ["↑ NORTH", "→ EAST", "↓ SOUTH", "← WEST"]
        for i, h in enumerate(heading_names):
            tk.Radiobutton(ctrl, text=h, variable=self._pol_heading, value=i,
                           command=self._refresh_policy_tab,
                           bg=THEME["panel"], fg=THEME["text"],
                           selectcolor=THEME["bg"], activebackground=THEME["panel"],
                           font=("Segoe UI", 9)).pack(side="left", padx=6)

        tk.Label(ctrl, text="Goal:", bg=THEME["panel"],
                 fg=THEME["text_dim"], font=("Segoe UI", 9)).pack(side="left", padx=(16, 4))
        self._pol_goal = tk.IntVar(value=0)
        for i in range(3):
            tk.Radiobutton(ctrl, text=f"G{i}", variable=self._pol_goal, value=i,
                           command=self._refresh_policy_tab,
                           bg=THEME["panel"], fg=THEME["text"],
                           selectcolor=THEME["bg"], activebackground=THEME["panel"],
                           font=("Segoe UI", 9)).pack(side="left", padx=4)

        self._fig_pol = Figure(figsize=(9, 2.2), facecolor=THEME["panel"])
        self._ax_pol  = self._fig_pol.add_subplot(111)
        self._fig_pol.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(self._fig_pol, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas_pol = canvas
        self._refresh_policy_tab()

    def _refresh_policy_tab(self, *_):
        ax  = self._ax_pol
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_color(THEME["border"])

        agent_name = self._sel_agent.get()
        agent = self._agents[agent_name]
        G = self.env.GRID_SIZE

        if not hasattr(agent, "Q"):
            ax.text(0.5, 0.5, f"{agent_name} không có Q-table",
                    transform=ax.transAxes, ha="center", va="center",
                    color=THEME["text_dim"], fontsize=10)
            self._canvas_pol.draw()
            return

        heading = self._pol_heading.get()
        goal_id = self._pol_goal.get()

        ACTION_ARROWS = {
            self.env.FORWARD:     "↑↓←→",  # sẽ xử lý riêng
            self.env.TURN_LEFT:  "↺",
            self.env.TURN_RIGHT: "↻",
            self.env.STOP:       "■",
        }
        FWD_ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}

        for r in range(G):
            for c in range(G):
                if (r, c) in self.env.OBSTACLES:
                    color = "#374151"
                elif (r, c) == self.env.GOALS[goal_id]:
                    color = "#065F46"
                elif (r, c) in self.env.GOALS:
                    color = "#D1FAE5"
                else:
                    s = self.env.state_encoder((r, c, heading, goal_id))
                    best = int(np.argmax(agent.Q[s]))
                    q    = float(agent.Q[s, best])
                    # Tô màu theo giá trị Q (gradient xanh đậm → nhạt)
                    norm = max(0, min(1, (q + 30) / 80))
                    r_ch = int(30 + norm * 20)
                    g_ch = int(40 + norm * 60)
                    b_ch = int(80 + norm * 100)
                    color = f"#{r_ch:02x}{g_ch:02x}{b_ch:02x}"

                rect = mpatches.FancyBboxPatch(
                    (c, G - 1 - r), 1, 1,
                    boxstyle="square,pad=0",
                    facecolor=color, edgecolor="#2a2a3e", linewidth=0.6,
                )
                ax.add_patch(rect)

                if (r, c) not in self.env.OBSTACLES:
                    if (r, c) == self.env.GOALS[goal_id]:
                        sym = "★"
                        col = "#6EE7B7"
                    elif (r, c) in self.env.GOALS:
                        sym = "☆"
                        col = "#065F46"
                    else:
                        s    = self.env.state_encoder((r, c, heading, goal_id))
                        best = int(np.argmax(agent.Q[s]))
                        q    = float(agent.Q[s, best])
                        sym  = (FWD_ARROWS[heading] if best == self.env.FORWARD
                                else ACTION_ARROWS[best])
                        col  = AGENT_COLORS.get(agent_name, "white")
                    ax.text(c + 0.5, G - 1 - r + 0.58, sym,
                            ha="center", va="center", fontsize=13,
                            color=col, fontweight="bold")
                    if (r, c) not in self.env.GOALS:
                        ax.text(c + 0.5, G - 1 - r + 0.2,
                                f"{q:.0f}",
                                ha="center", va="center", fontsize=6,
                                color="#94A3B8")

        ax.set_xlim(0, G); ax.set_ylim(0, G)
        ax.set_xticks(np.arange(0.5, G)); ax.set_yticks(np.arange(0.5, G))
        ax.set_xticklabels(range(G), fontsize=7, color=THEME["text_dim"])
        ax.set_yticklabels(list(reversed(range(G))), fontsize=7, color=THEME["text_dim"])
        ax.tick_params(length=0)
        ax.set_aspect("equal")
        ax.set_title(
            f"Policy – {agent_name}  |  Heading: {self.env.HEADING_NAMES[heading]}  |  Goal: G{goal_id}",
            color=THEME["text"], fontsize=9, pad=4)

        self._fig_pol.tight_layout(pad=0.5)
        self._canvas_pol.draw()

    # ── Tab: Comparison ──────────────────────────────────────────────

    def _build_comparison_tab(self, parent):
        self._fig_cmp = Figure(figsize=(9, 2.4), facecolor=THEME["panel"])
        axes = [self._fig_cmp.add_subplot(1, 3, i+1) for i in range(3)]
        self._ax_cmp  = axes
        self._fig_cmp.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(self._fig_cmp, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas_cmp = canvas
        self._draw_comparison()

    def _draw_comparison(self):
        root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_path = os.path.join(root, "experiments", "results",
                                 "eval_summary_all.json")

        if not os.path.exists(eval_path):
            for ax in self._ax_cmp:
                ax.set_facecolor("#1a1a2e")
                ax.text(0.5, 0.5, "Chạy  python main.py evaluate  trước",
                        transform=ax.transAxes, ha="center", va="center",
                        color=THEME["text_dim"], fontsize=8)
            self._canvas_cmp.draw()
            return

        with open(eval_path, encoding="utf-8") as f:
            sums = [s for s in json.load(f) if s]

        metrics = [
            ("success_mean",  "success_std",  "Success Rate",    True),
            ("steps_mean",    "steps_std",    "Avg Steps",       False),
            ("collision_mean","collision_std","Collision Rate",  True),
        ]

        for ax, (mk, sk, title, is_pct) in zip(self._ax_cmp, metrics):
            ax.clear()
            ax.set_facecolor("#1a1a2e")
            for sp in ax.spines.values(): sp.set_color(THEME["border"])
            ax.tick_params(colors=THEME["text_dim"], labelsize=7)

            names  = [s["agent"].replace("_"," ").title() for s in sums]
            means  = [s[mk] for s in sums]
            stds   = [s[sk] for s in sums]
            colors = [AGENT_COLORS.get(
                s["agent"].replace("_","-").replace(" ","-").title(),
                "#94A3B8") for s in sums]

            bars = ax.bar(names, means, yerr=stds, capsize=4,
                          color=colors, alpha=0.85, edgecolor="none",
                          error_kw=dict(ecolor=THEME["text_dim"], lw=1))
            if is_pct:
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            for bar, m in zip(bars, means):
                label = f"{m:.1%}" if is_pct else f"{m:.1f}"
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(stds)*0.05,
                        label, ha="center", va="bottom",
                        color=THEME["text"], fontsize=7)

            ax.set_title(title, color=THEME["text"], fontsize=9, pad=4)
            ax.grid(axis="y", alpha=0.15, color=THEME["border"])

        self._fig_cmp.tight_layout(pad=1.2)
        self._canvas_cmp.draw()

    # ── Tab: Replay ──────────────────────────────────────────────────

    def _build_replay_tab(self, parent):
        self._fig_replay = Figure(figsize=(9, 2.4), facecolor=THEME["panel"])
        self._ax_replay  = self._fig_replay.add_subplot(111)
        self._fig_replay.tight_layout(pad=0.5)
        canvas = FigureCanvasTkAgg(self._fig_replay, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas_replay = canvas

        ctrl = tk.Frame(parent, bg=THEME["panel"])
        ctrl.pack(fill="x", padx=8, pady=(0, 4))
        tk.Button(ctrl, text="Lưu replay đường đi", font=("Segoe UI", 9),
                  bg=THEME["accent"], fg="white", relief="flat",
                  cursor="hand2", padx=8,
                  command=self._save_trail_plot).pack(side="left", pady=4)

        self._draw_replay()

    def _draw_replay(self):
        ax = self._ax_replay
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        G  = self.env.GRID_SIZE

        for r in range(G):
            for c in range(G):
                if (r, c) in self.env.OBSTACLES:
                    col = "#374151"
                elif (r, c) in self.env.GOALS:
                    col = "#065F46"
                else:
                    col = "#1a2a3e"
                ax.add_patch(mpatches.Rectangle(
                    (c, G-1-r), 1, 1,
                    facecolor=col, edgecolor="#2a2a3e", linewidth=0.5))

        if len(self._trail) > 1:
            xs = [c + 0.5 for r, c in self._trail]
            ys = [G-1-r + 0.5 for r, c in self._trail]
            ax.plot(xs, ys, color=AGENT_COLORS.get(self._sel_agent.get(), "white"),
                    linewidth=2, alpha=0.8, zorder=3)
            ax.scatter(xs[0],  ys[0],  s=80, color="#3B82F6",  zorder=4)
            ax.scatter(xs[-1], ys[-1], s=80, color="#10B981",  zorder=4)

        for i, (gr, gc) in enumerate(self.env.GOALS):
            ax.text(gc+0.5, G-1-gr+0.5, f"G{i}",
                    ha="center", va="center", fontsize=9,
                    color="#6EE7B7", fontweight="bold")

        ax.set_xlim(0, G); ax.set_ylim(0, G); ax.set_aspect("equal")
        ax.set_xticks(np.arange(0.5, G)); ax.set_yticks(np.arange(0.5, G))
        ax.set_xticklabels(range(G), fontsize=7, color=THEME["text_dim"])
        ax.set_yticklabels(list(reversed(range(G))), fontsize=7, color=THEME["text_dim"])
        ax.tick_params(length=0)
        ax.set_title(
            f"Đường đi – {self._sel_agent.get()} | "
            f"{len(self._trail)} bước",
            color=THEME["text"], fontsize=9, pad=4)
        for sp in ax.spines.values(): sp.set_color(THEME["border"])
        self._fig_replay.tight_layout(pad=0.5)
        self._canvas_replay.draw()

    def _save_trail_plot(self):
        root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path  = os.path.join(root, "reports", "figures", "last_trail.png")
        self._fig_replay.savefig(path, dpi=150, facecolor=THEME["panel"])
        self._set_status(f"Đã lưu: {path}")

    # ------------------------------------------------------------------ #
    #  Grid vẽ                                                             #
    # ------------------------------------------------------------------ #

    def _draw_grid(self):
        ax = self._ax_grid
        ax.clear()
        ax.set_facecolor(THEME["panel"])

        if self.env.state is None:
            return

        G = self.env.GRID_SIZE
        r, c, heading, goal_id = self.env.state
        goal_r, goal_c = self.env.GOALS[goal_id]

        # Cells
        for ri in range(G):
            for ci in range(G):
                # Màu cell
                if (ri, ci) in self.env.OBSTACLES:
                    col = CELL_OBS
                elif (ri, ci) == self.env.GOALS[goal_id]:
                    col = CELL_GOAL_ON
                elif (ri, ci) in self.env.GOALS:
                    col = CELL_GOAL_OFF
                elif len(self._trail) > 1 and (ri, ci) in self._trail[:-1]:
                    col = CELL_TRAIL
                else:
                    col = CELL_EMPTY

                ax.add_patch(mpatches.FancyBboxPatch(
                    (ci + 0.03, G-1-ri + 0.03), 0.94, 0.94,
                    boxstyle="round,pad=0.05",
                    facecolor=col, edgecolor="#CCCCCC",
                    linewidth=0.5, zorder=1))

        # Goal labels
        for i, (gr, gc) in enumerate(self.env.GOALS):
            sym   = "★" if i == goal_id else "☆"
            color = "white" if i == goal_id else "#2E7D32"
            ax.text(gc+0.5, G-1-gr+0.5, f"{sym}G{i}",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color, zorder=3)

        # Obstacle symbol
        for (or_, oc) in self.env.OBSTACLES:
            ax.text(oc+0.5, G-1-or_+0.5, "✕",
                    ha="center", va="center", fontsize=11,
                    color="white", fontweight="bold", zorder=3)

        # Trail line
        if len(self._trail) > 1:
            xs = [ci+0.5 for (_, ci) in self._trail]
            ys = [G-1-ri+0.5 for (ri, _) in self._trail]
            ax.plot(xs, ys, color="#1565C0", linewidth=1.8,
                    linestyle="--", alpha=0.5, zorder=2)

        # Car body
        circle = plt.Circle((c+0.5, G-1-r+0.5), 0.30,
                             facecolor=CELL_CAR, edgecolor="#B71C1C",
                             linewidth=1.5, zorder=5)
        ax.add_patch(circle)

        # Car arrow (heading)
        angles = {0: 90, 1: 0, 2: 270, 3: 180}
        angle  = angles[heading]
        dx = 0.32 * np.cos(np.radians(angle))
        dy = 0.32 * np.sin(np.radians(angle))
        ax.annotate("",
                    xy=(c+0.5+dx, G-1-r+0.5+dy),
                    xytext=(c+0.5-dx*0.3, G-1-r+0.5-dy*0.3),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                   lw=2, mutation_scale=16),
                    zorder=6)

        # Grid lines
        for i in range(G+1):
            ax.axhline(i, color="#AAAAAA", linewidth=0.3, alpha=0.5)
            ax.axvline(i, color="#AAAAAA", linewidth=0.3, alpha=0.5)

        # Axis labels
        ax.set_xlim(0, G); ax.set_ylim(0, G)
        ax.set_xticks(np.arange(0.5, G))
        ax.set_yticks(np.arange(0.5, G))
        ax.set_xticklabels(range(G), fontsize=8, color="#666")
        ax.set_yticklabels(list(reversed(range(G))), fontsize=8, color="#666")
        ax.tick_params(length=0)
        ax.set_aspect("equal")

        dist = abs(r - goal_r) + abs(c - goal_c)
        ax.set_title(
            f"{self._sel_agent.get()}  |  "
            f"({r},{c}) {self.env.HEADING_ARROWS[heading]}  |  "
            f"dist={dist}  |  step={self.env.steps}",
            fontsize=9, pad=5)

        self._fig_grid.tight_layout(pad=0.5)
        self._canvas_grid.draw()

    # ------------------------------------------------------------------ #
    #  Info panel update                                                   #
    # ------------------------------------------------------------------ #

    def _update_info(self):
        if self.env.state is None:
            return
        r, c, heading, goal_id = self.env.state
        goal_r, goal_c = self.env.GOALS[goal_id]
        dist = abs(r - goal_r) + abs(c - goal_c)

        an = (self.env.ACTION_NAMES.get(self._last_action, "–")
              if self._last_action is not None else "–")

        self._info_vars["pos"].set(f"({r}, {c})")
        self._info_vars["heading"].set(
            f"{self.env.HEADING_ARROWS[heading]} {self.env.HEADING_NAMES[heading]}")
        self._info_vars["goal"].set(f"G{goal_id}  {self.env.GOALS[goal_id]}")
        self._info_vars["dist"].set(str(dist))
        self._info_vars["action"].set(an)
        self._info_vars["reward"].set(f"{self._last_reward:+.0f}")
        self._info_vars["total"].set(f"{self._ep_reward:+.1f}")
        self._info_vars["steps"].set(str(self.env.steps))

        if self._done:
            if self._last_reward and self._last_reward >= 50:
                self._info_vars["status"].set("✅ THÀNH CÔNG!")
            elif self._last_reward and self._last_reward <= -30:
                self._info_vars["status"].set("💥 VA CHẠM!")
            else:
                self._info_vars["status"].set("⏰ Hết bước")
        else:
            self._info_vars["status"].set("▶ Đang chạy")

        # Reward bar
        w = self._bar_canvas.winfo_width() or 160
        max_r = 200
        norm  = max(0, min(1, (self._ep_reward + max_r) / (2 * max_r)))
        color = THEME["success"] if self._ep_reward >= 0 else THEME["danger"]
        self._bar_canvas.delete("all")
        self._bar_canvas.create_rectangle(
            0, 0, int(w * norm), 16, fill=color, outline="")
        self._bar_canvas.create_text(
            w//2, 8, text=f"{self._ep_reward:+.0f}",
            fill="white", font=("Consolas", 8))

    # ------------------------------------------------------------------ #
    #  Episode management                                                  #
    # ------------------------------------------------------------------ #

    def _reset_episode(self):
        self._stop_auto()
        self._trail         = []
        self._ep_reward     = 0.0
        self._last_action   = None
        self._last_reward   = 0.0
        self._done          = False

        self._obs, _ = self.env.reset()
        # Ép goal_id theo selector
        r, c, h, _ = self.env.state_decoder(self._obs)
        goal_id     = self._sel_goal.get()
        self.env._state = (r, c, h, goal_id)
        self._obs = self.env.state_encoder(self.env._state)

        self._trail.append((r, c))
        self._redraw()
        self._set_status(
            f"Sẵn sàng – thuật toán [{self._sel_agent.get()}] | "
            f"nhấn ▶ CHẠY để bắt đầu")

    def _do_step(self):
        if self._done:
            self._reset_episode()
            return

        agent = self._agents.get(self._sel_agent.get())
        name  = self._sel_agent.get()

        # Thuật toán chưa được cài đặt → không chạy, báo nhẹ nhàng
        if agent is None:
            self._stop_auto()
            self._set_status(
                f"⚠ Thuật toán [{name}] chưa được cài đặt "
                f"(agents/{name.lower().replace('-', '_')}.py).")
            return

        try:
            action = agent.select_action(self._obs, eval_mode=True)
        except TypeError:
            # Agent không nhận eval_mode (vd RandomAgent)
            try:
                action = agent.select_action(self._obs)
            except NotImplementedError:
                self._stop_auto()
                self._set_status(f"⚠ Thuật toán [{name}] chưa được cài đặt.")
                return
        except NotImplementedError:
            self._stop_auto()
            self._set_status(f"⚠ Thuật toán [{name}] chưa được cài đặt.")
            return

        next_obs, reward, terminated, truncated, info = self.env.step(action)

        self._last_action  = action
        self._last_reward  = reward
        self._ep_reward   += reward
        self._obs          = next_obs
        self._done         = terminated or truncated

        r, c, _, _ = self.env.state_decoder(self._obs)
        self._trail.append((r, c))

        if self._done:
            self._ep_count += 1
            if info.get("reached_goal"):
                self._wins += 1
            self._stat_ep.set(f"Episode: {self._ep_count}")
            self._stat_win.set(f"Thành công: {self._wins}")
            rate = self._wins / self._ep_count if self._ep_count else 0
            self._stat_rate.set(f"Tỷ lệ: {rate:.1%}")

        self._redraw()
        status = (f"Action: {self.env.ACTION_NAMES.get(action,'?')}  |  "
                  f"Reward: {reward:+.0f}  |  Total: {self._ep_reward:+.1f}")
        self._set_status(status)

    def _redraw(self):
        self._draw_grid()
        self._update_info()
        self._draw_replay()
        # Nếu tab Policy đang hiển thị, refresh
        try:
            if self._nb.index("current") == 1:
                self._refresh_policy_tab()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Auto-run                                                            #
    # ------------------------------------------------------------------ #

    def _on_auto(self):
        agent = self._agents.get(self._sel_agent.get())
        name  = self._sel_agent.get()
        if agent is None:
            self._set_status(
                f"⚠ Thuật toán [{name}] chưa được cài đặt "
                f"(agents/{name.lower().replace('-', '_')}.py) – không thể chạy.")
            return
        if self._done:
            self._reset_episode()
        self._btn_auto.config(bg=THEME["accent"], text=f"⏩  Đang chạy [{name}]...")
        self._set_status(f"▶ Đang chạy thuật toán: {name}")
        self._auto_running = True
        self._auto_loop()

    def _auto_loop(self):
        if not self._auto_running:
            return
        if self._done:
            self._btn_auto.config(bg=THEME["success"], text="▶  CHẠY thuật toán")
            self._auto_running = False
            return
        self._do_step()
        # Nếu _do_step đã dừng auto (vd thuật toán chưa cài) thì không lặp tiếp
        if not self._auto_running:
            return
        speed = self._auto_speed.get()
        self._auto_id = self.after(speed, self._auto_loop)

    def _stop_auto(self):
        self._auto_running = False
        if self._auto_id:
            self.after_cancel(self._auto_id)
            self._auto_id = None
        self._btn_auto.config(bg=THEME["success"], text="▶  CHẠY thuật toán")

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _on_step(self):
        self._stop_auto()
        self._do_step()

    def _on_reset(self):
        self._reset_episode()

    def _on_agent_change(self):
        self._stop_auto()
        self._reset_episode()
        self._refresh_policy_tab()

    def _on_goal_change(self):
        self._stop_auto()
        self._reset_episode()

    # ------------------------------------------------------------------ #
    #  Đổi bản đồ / độ khó                                                 #
    # ------------------------------------------------------------------ #

    def _on_change_map(self):
        """Đổi bản đồ theo độ khó đang chọn và train lại RL agents."""
        self._stop_auto()
        choice = self._sel_difficulty.get()

        if choice == "Cố định":
            self.env.use_fixed_map()
            self._reload_fixed_agents()
            self._set_status("Bản đồ cố định – đã nạp lại Q-table đã huấn luyện.")
        else:
            # Sinh bản đồ ngẫu nhiên mới (đảm bảo đi được) theo độ khó
            self.env.regenerate_map(difficulty=choice)
            n_obs = len(self.env.OBSTACLES)
            self._set_status(
                f"Đang train lại trên bản đồ [{choice}] – {n_obs} vật cản...")
            self.update_idletasks()
            self._train_rl_on_env()
            self._set_status(
                f"✅ Bản đồ [{choice}] – {n_obs} vật cản | đã train lại "
                f"Q-Learning/SARSA trên bản đồ này.")

        # Reset thống kê (đổi bản đồ → episode count cũ không còn ý nghĩa)
        self._ep_count = 0
        self._wins = 0
        self._stat_ep.set("Episode: 0")
        self._stat_win.set("Thành công: 0")
        self._stat_rate.set("Tỷ lệ: –")

        self._reset_episode()
        self._refresh_policy_tab()

    def _reload_fixed_agents(self):
        """Nạp lại Q-table đã train (map cố định) từ thư mục results."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(root, "experiments", "results")
        for key, fname, Cls in [
            ("Q-Learning", "q_learning", QLearningAgent),
            ("SARSA",      "sarsa",      SARSAAgent),
        ]:
            try:
                ag = Cls(self.env.n_states, self.env.n_actions,
                         alpha=0.1, gamma=0.95, seed=42)
            except Exception:
                self._agents[key] = None
                continue
            p = os.path.join(save_dir, f"{fname}_seed0_qtable.npy")
            if os.path.exists(p):
                try:
                    ag.load(p)
                    ag.epsilon = 0.0
                except Exception:
                    pass
            self._agents[key] = ag

    def _train_rl_on_env(self, n_ep: int = None, progress: bool = True):
        """
        Quick-train Q-Learning & SARSA trực tiếp trên self.env hiện tại.

        Siêu tham số (alpha, gamma, epsilon, n_episodes) lấy từ configs.yaml
        thông qua self._cfg → khớp với train chính (experiments/train.py).
        Tham số n_ep (nếu truyền) sẽ ghi đè n_episodes trong config, hữu ích
        khi muốn train nhanh hơn trong lúc demo tương tác.

        progress=True → cập nhật thanh trạng thái + in console theo thời gian
        thực để người dùng theo dõi quá trình học (thay vì chạy ẩn).
        """
        from experiments.train import run_episode_qlearning, run_episode_sarsa
        specs = [
            ("Q-Learning", QLearningAgent, run_episode_qlearning, "q_learning"),
            ("SARSA",      SARSAAgent,     run_episode_sarsa,     "sarsa"),
        ]
        for key, Cls, fn, cfg_key in specs:
            hp = _hp(self._cfg, cfg_key)
            episodes = n_ep if n_ep is not None else hp["n_episodes"]
            try:
                ag = Cls(self.env.n_states, self.env.n_actions,
                         alpha=hp["alpha"], gamma=hp["gamma"],
                         epsilon_start=hp["epsilon_start"],
                         epsilon_end=hp["epsilon_end"],
                         epsilon_decay=hp["epsilon_decay"], seed=0)

                # Cập nhật tiến trình ~100 lần trong suốt quá trình train
                log_every = max(1, episodes // 100)
                recent = []          # cửa sổ success gần đây (tối đa 100)
                n_success = 0

                for ep in range(episodes):
                    res = fn(ag, self.env)
                    ag.decay_epsilon()

                    s = int(res.get("success", 0))
                    n_success += s
                    recent.append(s)
                    if len(recent) > 100:
                        recent.pop(0)

                    if progress and (ep % log_every == 0 or ep == episodes - 1):
                        sr = sum(recent) / len(recent) if recent else 0.0
                        pct = (ep + 1) / episodes
                        msg = (f"🧠 Train {key}: ep {ep+1}/{episodes} "
                               f"({pct:.0%}) | ε={ag.epsilon:.3f} | "
                               f"success~{sr:.0%}")
                        self._set_status(msg)
                        # Vẽ thanh tiến trình ngay trên grid để dễ thấy
                        self._draw_train_progress(key, pct, sr, ag.epsilon)
                        self.update()    # repaint UI ngay lập tức

                ag.epsilon = 0.0
                self._agents[key] = ag
                if progress:
                    final_sr = n_success / episodes if episodes else 0.0
                    print(f"[TRAIN] {key}: {episodes} ep | "
                          f"tổng success={final_sr:.1%}")
            except NotImplementedError:
                # Thuật toán chưa cài đặt → bỏ qua, giữ nguyên trạng thái cũ
                if progress:
                    print(f"[TRAIN] {key}: chưa cài đặt → bỏ qua.")
                continue
            except Exception as e:
                print(f"[WARN] Train lại {key} thất bại: {e}")
                continue

    def _draw_train_progress(self, agent_name: str, pct: float,
                             success_rate: float, epsilon: float):
        """Vẽ overlay tiến trình train lên canvas grid (thanh % + chỉ số)."""
        ax = self._ax_grid
        ax.clear()
        ax.set_facecolor(THEME["panel"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

        color = AGENT_COLORS.get(agent_name, THEME["accent"])

        ax.text(0.5, 0.78, f"🧠 Đang huấn luyện {agent_name}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color=THEME["text"], transform=ax.transAxes)

        # Khung thanh tiến trình
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, 0.48), 0.8, 0.1, boxstyle="round,pad=0.01",
            facecolor="#1a1a2e", edgecolor=THEME["border"], linewidth=1.2,
            transform=ax.transAxes))
        # Phần đã hoàn thành
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, 0.48), max(0.001, 0.8 * pct), 0.1, boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="none", transform=ax.transAxes))
        ax.text(0.5, 0.53, f"{pct:.0%}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                transform=ax.transAxes)

        ax.text(0.5, 0.30, f"Tỉ lệ thành công (gần đây): {success_rate:.0%}",
                ha="center", va="center", fontsize=10,
                color=THEME["success"] if success_rate >= 0.85 else THEME["warning"],
                transform=ax.transAxes)
        ax.text(0.5, 0.18, f"Epsilon (mức khám phá): {epsilon:.3f}",
                ha="center", va="center", fontsize=10,
                color=THEME["text_dim"], transform=ax.transAxes)

        self._canvas_grid.draw()

    def _on_close(self):
        self._stop_auto()
        plt.close("all")
        self.destroy()

    # ------------------------------------------------------------------ #
    #  Tiện ích                                                            #
    # ------------------------------------------------------------------ #

    def _set_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _center_window(self, w: int, h: int):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - w) // 2
        y  = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")


# ======================================================================= #
#  CLI                                                                      #
# ======================================================================= #

def main():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Dashboard Demo – Đề tài 15")
    parser.add_argument("--train_first", action="store_true",
                        help="Train nhanh 3000 ep rồi mở dashboard")
    args = parser.parse_args()

    pretrained = None
    if args.train_first:
        cfg = load_train_config()
        n_ql = _hp(cfg, "q_learning")["n_episodes"]
        print(f"Training nhanh Q-Learning & SARSA ({n_ql} episodes theo configs.yaml)...")
        pretrained = quick_train(cfg=cfg)
        print("Xong. Mở dashboard...\n")

    app = App(pretrained=pretrained)
    app.mainloop()


if __name__ == "__main__":
    main()
