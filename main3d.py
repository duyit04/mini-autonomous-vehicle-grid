"""
main3d.py  –  Trình mô phỏng 3D cho Xe tự hành mini (Đề tài 15)
================================================================
Phiên bản 3D ĐỘC LẬP của bản đồ grid 7×7. Xe chạy trên ma trận 3D,
có thể XOAY camera quanh bản đồ như game (kéo chuột hoặc tự động xoay).

File này KHÔNG ảnh hưởng tới code hiện tại:
  - Tái sử dụng nguyên môi trường envs.custom_env.DirectionalCarEnv
  - Tự load Q-table đã train (experiments/results/*_qtable.npy) nếu có
  - Policy được tính nội bộ (Q-Learning / SARSA / Heuristic / Random)
    nên chạy được kể cả khi các agent stub chưa cài đặt.

Chạy:
    python main3d.py
    python main.py demo3d           (nếu đã thêm lệnh – tùy chọn)

Điều khiển:
    - Kéo chuột trái     : xoay camera quanh bản đồ
    - Cuộn chuột         : zoom
    - Nút ▶ CHẠY         : cho xe tự chạy theo thuật toán đang chọn
    - ☑ Tự xoay camera   : camera quay vòng quanh bản đồ như game
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from envs.custom_env import DirectionalCarEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSAAgent


# ── Bảng màu (đồng bộ với dashboard 2D) ────────────────────────────────
THEME = {
    "bg":        "#0F1021",
    "panel":     "#1B1C30",
    "accent":    "#7C3AED",
    "accent2":   "#06B6D4",
    "success":   "#10B981",
    "danger":    "#EF4444",
    "warning":   "#F59E0B",
    "text":      "#E2E8F0",
    "text_dim":  "#94A3B8",
    "border":    "#33344F",
}

AGENT_COLORS = {
    "Q-Learning": "#3B82F6",
    "SARSA":      "#10B981",
    "Heuristic":  "#F59E0B",
    "Random":     "#94A3B8",
}

# Màu scene 3D
C_FLOOR        = "#262A4A"
C_FLOOR_ALT    = "#2E3358"
C_FLOOR_EDGE   = "#3D4270"
C_OBSTACLE     = "#475569"
C_OBSTACLE_TOP = "#64748B"
C_GOAL_ON      = "#10B981"
C_GOAL_OFF     = "#3F6B5C"
C_TRAIL        = "#38BDF8"
C_CAR_BODY     = "#EF4444"
C_CAR_CABIN    = "#FCA5A5"
C_NOSE         = "#FACC15"


# ======================================================================= #
#  Policy nội bộ – không phụ thuộc vào các agent stub                       #
# ======================================================================= #

class Policy:
    """Bọc một cách chọn action thống nhất: select(state) -> action."""

    def __init__(self, name, env):
        self.name = name
        self.env = env
        self.q_table = None
        self.rng = np.random.default_rng(0)
        self.available = True
        self.note = ""

    # ---- nạp Q-table (.npy) cho Q-Learning / SARSA --------------------
    def load_qtable(self, path):
        try:
            q = np.load(path)
            if q.shape == (self.env.n_states, self.env.n_actions):
                self.q_table = q
                self.note = "đã load Q-table"
                return True
        except Exception as e:
            self.note = f"lỗi load: {e.__class__.__name__}"
        return False

    # ---- chọn action --------------------------------------------------
    def select(self, state):
        if self.name in ("Q-Learning", "SARSA"):
            if self.q_table is None:
                # fallback heuristic nếu chưa train
                return self._heuristic(state)
            row = self.q_table[state]
            best = np.flatnonzero(np.isclose(row, row.max()))
            return int(best[0])
        if self.name == "Heuristic":
            return self._heuristic(state)
        # Random
        return int(self.rng.integers(self.env.n_actions))

    # ---- heuristic greedy hướng tới goal ------------------------------
    def _heuristic(self, state):
        env = self.env
        r, c, heading, goal_id = env.state_decoder(state)
        gr, gc = env.GOALS[goal_id]

        if (r, c) == (gr, gc):
            return env.STOP

        # hướng mong muốn: ưu tiên trục còn xa hơn
        if abs(gr - r) >= abs(gc - c) and gr != r:
            desired = env.NORTH if gr < r else env.SOUTH
        elif gc != c:
            desired = env.EAST if gc > c else env.WEST
        else:
            desired = env.NORTH if gr < r else env.SOUTH

        if heading == desired:
            dr, dc = env.DELTAS[heading]
            if env.is_valid_cell(r + dr, c + dc):
                return env.FORWARD
            return env.TURN_RIGHT  # bị chặn → xoay tìm đường

        return env.TURN_RIGHT if (desired - heading) % 4 == 1 else env.TURN_LEFT


# ======================================================================= #
#  Tiện ích hình học 3D                                                     #
# ======================================================================= #

def cuboid_faces(x0, y0, z0, dx, dy, dz):
    """Trả về 6 mặt (mỗi mặt 4 đỉnh) của khối hộp chữ nhật."""
    x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    return [
        [v[0], v[1], v[2], v[3]],  # đáy
        [v[4], v[5], v[6], v[7]],  # nóc
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[0], v[3], v[7], v[4]],
    ]


# ======================================================================= #
#  Ứng dụng chính                                                           #
# ======================================================================= #

class CarViz3D(tk.Tk):

    AGENTS = ["Q-Learning", "SARSA", "Heuristic", "Random"]
    GOALS  = ["G0 – (0, 6)", "G1 – (3, 6)", "G2 – (6, 0)"]

    def __init__(self):
        super().__init__()
        self.title("Đề tài 15 – Xe tự hành mini 3D")
        self.configure(bg=THEME["bg"])
        self.minsize(1080, 700)
        self._center(1220, 760)

        self.env = DirectionalCarEnv(max_steps=200)
        self.G = self.env.GRID_SIZE
        self._policies = self._init_policies()

        # biến điều khiển
        self._sel_agent = tk.StringVar(value="Q-Learning")
        self._sel_goal  = tk.IntVar(value=0)
        self._speed     = tk.IntVar(value=220)
        self._autorot   = tk.BooleanVar(value=False)

        # trạng thái episode
        self._obs        = None
        self._trail      = []
        self._ep_reward  = 0.0
        self._last_action = None
        self._last_reward = 0.0
        self._done        = False
        self._ep_count    = 0
        self._wins        = 0
        self._auto_id     = None
        self._auto_on     = False
        self._azim        = -60.0
        self._elev        = 30.0
        self._dragging    = False

        self._build_ui()
        self._reset_episode()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick_rotation()

    # ------------------------------------------------------------------ #
    #  Khởi tạo policies + load Q-table                                    #
    # ------------------------------------------------------------------ #

    def _init_policies(self):
        save_dir = os.path.join(ROOT, "experiments", "results")
        pol = {name: Policy(name, self.env) for name in self.AGENTS}
        for name, fname in [("Q-Learning", "q_learning"), ("SARSA", "sarsa")]:
            p = os.path.join(save_dir, f"{fname}_seed0_qtable.npy")
            if os.path.exists(p):
                pol[name].load_qtable(p)
        return pol

    # ------------------------------------------------------------------ #
    #  UI                                                                  #
    # ------------------------------------------------------------------ #

    def _center(self, w, h):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//3}")

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=THEME["accent"], height=44)
        hdr.pack(fill="x", side="top")
        tk.Label(hdr, text="🚗  Xe Tự Hành Mini  –  Bản đồ 3D  –  Đề tài 15",
                 bg=THEME["accent"], fg="white",
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=16, pady=8)
        tk.Label(hdr, text="Kéo chuột để xoay • Cuộn để zoom",
                 bg=THEME["accent"], fg="#DDD6FE",
                 font=("Segoe UI", 9)).pack(side="right", padx=16)

        body = tk.Frame(self, bg=THEME["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=THEME["panel"], width=220)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 6))
        left.pack_propagate(False)
        self._build_controls(left)

        center = tk.Frame(body, bg=THEME["panel"],
                          highlightthickness=1, highlightbackground=THEME["border"])
        center.grid(row=0, column=1, sticky="nsew")
        self._build_canvas(center)

        right = tk.Frame(body, bg=THEME["panel"], width=210,
                         highlightthickness=1, highlightbackground=THEME["border"])
        right.grid(row=0, column=2, sticky="ns", padx=(6, 0))
        right.pack_propagate(False)
        self._build_info(right)

        self._status = tk.StringVar(value="Sẵn sàng")
        tk.Label(self, textvariable=self._status, bg=THEME["border"],
                 fg=THEME["text_dim"], font=("Consolas", 9),
                 anchor="w", padx=10).pack(fill="x", side="bottom")

    def _sep(self, parent):
        tk.Frame(parent, bg=THEME["border"], height=1).pack(fill="x", padx=8, pady=5)

    def _build_controls(self, parent):
        tk.Label(parent, text="⚙ ĐIỀU KHIỂN", bg=THEME["panel"],
                 fg=THEME["accent2"], font=("Segoe UI", 10, "bold")
                 ).pack(padx=12, pady=8, anchor="w")
        self._sep(parent)

        tk.Label(parent, text="Thuật toán:", bg=THEME["panel"], fg=THEME["text"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(6, 2), anchor="w")
        for name in self.AGENTS:
            color = AGENT_COLORS[name]
            note = self._policies[name].note
            txt = name + ("  ✓" if note.startswith("đã load") else "")
            rb = tk.Radiobutton(
                parent, text=txt, variable=self._sel_agent, value=name,
                command=self._on_agent_change,
                bg=THEME["panel"], fg=THEME["text"], selectcolor=THEME["bg"],
                activebackground=THEME["panel"], activeforeground=color,
                indicatoron=0, relief="flat", bd=0, font=("Segoe UI", 9),
                cursor="hand2", padx=10, pady=4, width=18, anchor="w")
            rb.pack(padx=12, pady=1, fill="x")
            rb.bind("<Enter>", lambda e, r=rb, c=color: r.config(fg=c))
            rb.bind("<Leave>", lambda e, r=rb: r.config(fg=THEME["text"]))

        self._sep(parent)
        tk.Label(parent, text="Goal:", bg=THEME["panel"], fg=THEME["text"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(6, 2), anchor="w")
        for i, g in enumerate(self.GOALS):
            tk.Radiobutton(parent, text=g, variable=self._sel_goal, value=i,
                           command=self._on_goal_change,
                           bg=THEME["panel"], fg=THEME["text"], selectcolor=THEME["bg"],
                           activebackground=THEME["panel"], indicatoron=0,
                           relief="flat", bd=0, font=("Segoe UI", 9), cursor="hand2",
                           padx=10, pady=3, width=18, anchor="w").pack(
                padx=12, pady=1, fill="x")

        self._sep(parent)
        # Bản đồ / Độ khó (đồng bộ với dashboard 2D)
        tk.Label(parent, text="Bản đồ / Độ khó:", bg=THEME["panel"], fg=THEME["text"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(6, 2), anchor="w")
        self._sel_difficulty = tk.StringVar(value="Cố định")
        diff_options = ["Cố định", "easy", "medium", "hard", "extreme"]
        om = tk.OptionMenu(parent, self._sel_difficulty, *diff_options)
        om.config(bg=THEME["bg"], fg=THEME["text"], font=("Segoe UI", 9),
                  activebackground=THEME["accent"], activeforeground="white",
                  highlightthickness=0, relief="flat", cursor="hand2", width=14)
        om["menu"].config(bg=THEME["panel"], fg=THEME["text"])
        om.pack(padx=12, pady=2, fill="x")
        tk.Button(parent, text="🗺  Tạo bản đồ & train lại", bg=THEME["accent"],
                  fg="white", font=("Segoe UI", 9, "bold"), relief="flat", bd=0,
                  cursor="hand2", pady=6, command=self._on_change_map).pack(
            padx=12, pady=4, fill="x")

        self._sep(parent)
        tk.Label(parent, text="Tốc độ (ms/bước):", bg=THEME["panel"],
                 fg=THEME["text"], font=("Segoe UI", 9)).pack(padx=12, anchor="w")
        tk.Scale(parent, from_=60, to=800, variable=self._speed, orient="horizontal",
                 bg=THEME["panel"], fg=THEME["text"], troughcolor=THEME["bg"],
                 highlightthickness=0, bd=0, font=("Segoe UI", 8)).pack(padx=12, fill="x")

        tk.Checkbutton(parent, text="🎥 Tự xoay camera", variable=self._autorot,
                       bg=THEME["panel"], fg=THEME["text"], selectcolor=THEME["bg"],
                       activebackground=THEME["panel"], font=("Segoe UI", 9),
                       anchor="w").pack(padx=12, pady=(6, 2), fill="x")

        self._sep(parent)
        bcfg = dict(font=("Segoe UI", 10, "bold"), relief="flat", bd=0,
                    cursor="hand2", pady=8)
        tk.Button(parent, text="▶  CHẠY", bg=THEME["success"], fg="white",
                  command=self._on_auto, **bcfg).pack(padx=12, pady=4, fill="x")
        tk.Button(parent, text="⏭  Bước tiếp", bg=THEME["accent2"], fg="white",
                  command=self._on_step, **bcfg).pack(padx=12, pady=4, fill="x")
        tk.Button(parent, text="⏹  Dừng", bg=THEME["border"], fg=THEME["text"],
                  command=self._stop_auto, **bcfg).pack(padx=12, pady=4, fill="x")
        tk.Button(parent, text="↺  Episode mới", bg="#374151", fg=THEME["text"],
                  command=self._on_reset, **bcfg).pack(padx=12, pady=4, fill="x")

        self._sep(parent)
        tk.Label(parent, text="📈 THỐNG KÊ", bg=THEME["panel"], fg=THEME["accent2"],
                 font=("Segoe UI", 9, "bold")).pack(padx=12, pady=(6, 2), anchor="w")
        self._stat_ep   = tk.StringVar(value="Episode: 0")
        self._stat_win  = tk.StringVar(value="Thành công: 0")
        self._stat_rate = tk.StringVar(value="Tỷ lệ: –")
        for v in (self._stat_ep, self._stat_win, self._stat_rate):
            tk.Label(parent, textvariable=v, bg=THEME["panel"], fg=THEME["text_dim"],
                     font=("Consolas", 9)).pack(padx=12, anchor="w")

    def _build_canvas(self, parent):
        self._fig = Figure(figsize=(6.4, 6.0), facecolor=THEME["panel"])
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasTkAgg(self._fig, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        self._canvas = canvas
        # đặt góc nhìn ban đầu cố định, gọn gàng
        self._ax.view_init(elev=self._elev, azim=self._azim)
        # tạm dừng auto-xoay khi người dùng đang kéo chuột để xoay tay
        canvas.mpl_connect("button_press_event", self._on_mouse_down)
        canvas.mpl_connect("button_release_event", self._on_mouse_up)

    def _on_mouse_down(self, event):
        self._dragging = True

    def _on_mouse_up(self, event):
        self._dragging = False
        # ghi nhớ góc người dùng vừa xoay tới
        try:
            self._azim = float(self._ax.azim)
            self._elev = float(self._ax.elev)
        except Exception:
            pass

    def _build_info(self, parent):
        tk.Label(parent, text="ℹ  THÔNG TIN", bg=THEME["panel"], fg=THEME["accent2"],
                 font=("Segoe UI", 10, "bold")).pack(padx=10, pady=(10, 4), anchor="w")
        tk.Frame(parent, bg=THEME["border"], height=1).pack(fill="x", padx=8, pady=2)

        self._info = {}
        fields = [("agent", "Agent"), ("pos", "Vị trí"), ("heading", "Hướng"),
                  ("goal", "Goal"), ("dist", "Khoảng cách"), ("", ""),
                  ("action", "Hành động"), ("reward", "Reward"),
                  ("total", "Tổng reward"), ("steps", "Bước"), ("", ""),
                  ("status", "Trạng thái")]
        for key, label in fields:
            if not key:
                tk.Frame(parent, bg=THEME["border"], height=1).pack(
                    fill="x", padx=8, pady=3)
                continue
            row = tk.Frame(parent, bg=THEME["panel"])
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=f"{label}:", width=12, anchor="w", bg=THEME["panel"],
                     fg=THEME["text_dim"], font=("Segoe UI", 8)).pack(side="left")
            var = tk.StringVar(value="–")
            tk.Label(row, textvariable=var, anchor="w", bg=THEME["panel"],
                     fg=THEME["text"], font=("Consolas", 9)).pack(side="left")
            self._info[key] = var

        tk.Frame(parent, bg=THEME["border"], height=1).pack(fill="x", padx=8, pady=6)
        tk.Label(parent, text="Chú thích:", bg=THEME["panel"], fg=THEME["text_dim"],
                 font=("Segoe UI", 8, "bold")).pack(padx=10, anchor="w")
        legend = [("🟥 Xe", C_CAR_BODY), ("🟩 Goal đích", C_GOAL_ON),
                  ("⬛ Vật cản", C_OBSTACLE), ("🟦 Vết đi", C_TRAIL)]
        for txt, _ in legend:
            tk.Label(parent, text="• " + txt, bg=THEME["panel"], fg=THEME["text"],
                     font=("Segoe UI", 8)).pack(padx=12, anchor="w")

    # ------------------------------------------------------------------ #
    #  Logic episode                                                       #
    # ------------------------------------------------------------------ #

    def _reset_episode(self):
        self._stop_auto()
        self._obs, _ = self.env.reset(seed=np.random.randint(0, 10_000))
        # ép goal theo lựa chọn người dùng
        r, c, h, _ = self.env.state_decoder(self._obs)
        self.env._state = (r, c, h, self._sel_goal.get())
        self._obs = self.env.state_encoder(self.env._state)

        self._trail = [(r, c)]
        self._ep_reward = 0.0
        self._last_action = None
        self._last_reward = 0.0
        self._done = False
        self._render()
        self._update_info("Episode mới – sẵn sàng")

    def _do_step(self):
        if self._done:
            return
        policy = self._policies[self._sel_agent.get()]
        action = policy.select(self._obs)
        obs, reward, term, trunc, info = self.env.step(action)
        self._obs = obs
        self._ep_reward += reward
        self._last_action = action
        self._last_reward = reward

        r, c, _, _ = self.env.state_decoder(obs)
        if not self._trail or self._trail[-1] != (r, c):
            self._trail.append((r, c))

        if term or trunc:
            self._done = True
            self._ep_count += 1
            if info.get("reached_goal"):
                self._wins += 1
                msg = "✅ ĐẾN ĐÍCH!"
            elif info.get("collision"):
                msg = "❌ VA CHẠM!"
            else:
                msg = "⏰ Hết bước"
            self._update_info(msg)
            self._update_stats()
        else:
            self._update_info("Đang chạy…")

        self._render()

    # ---- nút bấm ------------------------------------------------------
    def _on_step(self):
        self._stop_auto()
        if self._done:
            self._reset_episode()
        self._do_step()

    def _on_auto(self):
        if self._done:
            self._reset_episode()
        if self._auto_on:
            return
        self._auto_on = True
        self._status.set("▶ Đang chạy tự động…")
        self._auto_loop()

    def _auto_loop(self):
        if not self._auto_on:
            return
        if self._done:
            # nghỉ 1 nhịp rồi tự bắt đầu episode mới (chạy liên tục như game)
            self._auto_id = self.after(900, self._auto_restart)
            return
        self._do_step()
        self._auto_id = self.after(self._speed.get(), self._auto_loop)

    def _auto_restart(self):
        if not self._auto_on:
            return
        self._reset_episode_keep_auto()
        self._auto_id = self.after(self._speed.get(), self._auto_loop)

    def _reset_episode_keep_auto(self):
        self._obs, _ = self.env.reset(seed=np.random.randint(0, 10_000))
        r, c, h, _ = self.env.state_decoder(self._obs)
        self.env._state = (r, c, h, self._sel_goal.get())
        self._obs = self.env.state_encoder(self.env._state)
        self._trail = [(r, c)]
        self._ep_reward = 0.0
        self._last_action = None
        self._last_reward = 0.0
        self._done = False
        self._render()

    def _stop_auto(self):
        self._auto_on = False
        if self._auto_id is not None:
            try:
                self.after_cancel(self._auto_id)
            except Exception:
                pass
            self._auto_id = None
        self._status.set("⏹ Đã dừng")

    def _on_reset(self):
        self._reset_episode()

    def _on_agent_change(self):
        self._reset_episode()

    def _on_goal_change(self):
        self._reset_episode()

    # ------------------------------------------------------------------ #
    #  Đổi bản đồ / độ khó (đồng bộ dashboard 2D)                          #
    # ------------------------------------------------------------------ #

    def _on_change_map(self):
        """Đổi bản đồ theo độ khó đang chọn và train lại policy RL."""
        self._stop_auto()
        choice = self._sel_difficulty.get()

        if choice == "Cố định":
            self.env.use_fixed_map()
            self._reload_fixed_qtables()
            self._status.set("Bản đồ cố định – đã nạp lại Q-table đã train.")
        else:
            self.env.regenerate_map(difficulty=choice)
            n_obs = len(self.env.OBSTACLES)
            self._status.set(f"Đang train lại trên bản đồ [{choice}] – "
                             f"{n_obs} vật cản…")
            self.update_idletasks()
            self._train_on_env(n_ep=4000)
            self._status.set(f"✅ Bản đồ [{choice}] – {n_obs} vật cản | "
                             f"đã train lại Q-Learning/SARSA.")

        # reset thống kê (đổi bản đồ → số liệu cũ không còn ý nghĩa)
        self._ep_count = 0
        self._wins = 0
        self._update_stats()
        self._reset_episode()

    def _reload_fixed_qtables(self):
        """Nạp lại Q-table map cố định từ thư mục results vào policy."""
        save_dir = os.path.join(ROOT, "experiments", "results")
        for name, fname in [("Q-Learning", "q_learning"), ("SARSA", "sarsa")]:
            pol = self._policies[name]
            pol.q_table = None
            pol.note = ""
            p = os.path.join(save_dir, f"{fname}_seed0_qtable.npy")
            if os.path.exists(p):
                pol.load_qtable(p)

    def _train_on_env(self, n_ep: int = 4000):
        """Train nhanh Q-Learning & SARSA trực tiếp trên self.env hiện tại,
        rồi nạp q_table vào policy 3D. Agent stub (NotImplementedError) bỏ qua."""
        from experiments.train import (run_episode_qlearning,
                                        run_episode_sarsa)
        specs = [
            ("Q-Learning", QLearningAgent, run_episode_qlearning),
            ("SARSA",      SARSAAgent,     run_episode_sarsa),
        ]
        for name, Cls, fn in specs:
            try:
                ag = Cls(self.env.n_states, self.env.n_actions,
                         alpha=0.1, gamma=0.95, epsilon_start=1.0,
                         epsilon_end=0.01, epsilon_decay=0.999, seed=0)
                for _ in range(n_ep):
                    fn(ag, self.env)
                    ag.decay_epsilon()
                ag.epsilon = 0.0
                self._policies[name].q_table = ag.q_table.copy()
                self._policies[name].note = "đã train lại"
            except NotImplementedError:
                # thuật toán chưa cài → fallback heuristic
                self._policies[name].q_table = None
                self._policies[name].note = "fallback heuristic"
            except Exception as e:
                print(f"[WARN] Train lại {name} thất bại: {e}")

    # ------------------------------------------------------------------ #
    #  Tự xoay camera                                                      #
    # ------------------------------------------------------------------ #

    def _tick_rotation(self):
        # chỉ tự xoay khi bật và người dùng KHÔNG đang kéo chuột
        if self._autorot.get() and not self._dragging and not self._done:
            self._azim = (self._azim + 0.8) % 360
            try:
                self._ax.view_init(elev=self._elev, azim=self._azim)
                self._canvas.draw_idle()
            except Exception:
                pass
        self.after(60, self._tick_rotation)

    # ------------------------------------------------------------------ #
    #  Render scene 3D                                                     #
    # ------------------------------------------------------------------ #

    def _cell_xy(self, r, c):
        """Tâm ô (r,c) trong toạ độ 3D (x=col, y=hàng đảo ngược)."""
        return c + 0.5, (self.G - 1 - r) + 0.5

    @staticmethod
    def _add_box(faces, fcols, ecols, x, y, z, dx, dy, dz, color, edge):
        fs = cuboid_faces(x, y, z, dx, dy, dz)
        faces.extend(fs)
        fcols.extend([color] * 6)
        ecols.extend([edge] * 6)

    @staticmethod
    def _add_quad(faces, fcols, ecols, quad, color, edge):
        faces.append(quad)
        fcols.append(color)
        ecols.append(edge)

    def _render(self):
        ax = self._ax
        # lưu góc nhìn hiện tại TRƯỚC khi clear (ax.clear sẽ reset view)
        try:
            self._azim = float(ax.azim)
            self._elev = float(ax.elev)
        except Exception:
            pass

        ax.clear()
        G = self.G
        env = self.env

        ax.set_facecolor(THEME["panel"])
        try:
            ax.xaxis.set_pane_color((0, 0, 0, 0))
            ax.yaxis.set_pane_color((0, 0, 0, 0))
            ax.zaxis.set_pane_color((0, 0, 0, 0))
        except Exception:
            pass
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()

        # === Gom TẤT CẢ mặt phẳng vào 1 collection duy nhất ===
        # (matplotlib 3D chỉ sắp xếp chiều sâu đúng trong cùng 1 collection)
        faces, fcols, ecols = [], [], []

        # ---- sàn bàn cờ ----
        for r in range(G):
            for c in range(G):
                x0, y0 = c, (G - 1 - r)
                quad = [(x0, y0, 0.0), (x0 + 1, y0, 0.0),
                        (x0 + 1, y0 + 1, 0.0), (x0, y0 + 1, 0.0)]
                col = C_FLOOR if (r + c) % 2 == 0 else C_FLOOR_ALT
                self._add_quad(faces, fcols, ecols, quad, col, C_FLOOR_EDGE)

        # ---- vết đi (ô vuông phẳng nổi nhẹ trên sàn) ----
        for (tr, tc) in self._trail[:-1] if len(self._trail) > 1 else []:
            x0, y0 = tc + 0.18, (G - 1 - tr) + 0.18
            quad = [(x0, y0, 0.02), (x0 + 0.64, y0, 0.02),
                    (x0 + 0.64, y0 + 0.64, 0.02), (x0, y0 + 0.64, 0.02)]
            self._add_quad(faces, fcols, ecols, quad, C_TRAIL, C_TRAIL)

        # ---- vật cản (toà nhà) ----
        for (orr, occ) in env.OBSTACLES:
            x0, y0 = occ, (G - 1 - orr)
            self._add_box(faces, fcols, ecols,
                          x0 + 0.08, y0 + 0.08, 0.0, 0.84, 0.84, 0.95,
                          C_OBSTACLE, C_OBSTACLE_TOP)

        # ---- goals (cột) ----
        active_goal = self._sel_goal.get()
        goal_labels = []
        for gid, (gr, gc) in enumerate(env.GOALS):
            x0, y0 = gc, (G - 1 - gr)
            on = (gid == active_goal)
            color = C_GOAL_ON if on else C_GOAL_OFF
            h = 1.25 if on else 0.55
            self._add_box(faces, fcols, ecols,
                          x0 + 0.30, y0 + 0.30, 0.0, 0.40, 0.40, h,
                          color, "#A7F3D0" if on else "#2F5147")
            goal_labels.append((gr, gc, gid, on, h))

        # ---- xe (gộp luôn vào collection) ----
        self._add_car_faces(faces, fcols, ecols)

        poly = Poly3DCollection(faces, facecolors=fcols, edgecolors=ecols,
                                linewidths=0.5)
        ax.add_collection3d(poly)

        # ---- nhãn goal (text vẽ chồng lên) ----
        for (gr, gc, gid, on, h) in goal_labels:
            cx, cy = self._cell_xy(gr, gc)
            ax.text(cx, cy, h + 0.2, f"G{gid}",
                    color="#D1FAE5" if on else "#6B8F7E",
                    ha="center", va="bottom",
                    fontsize=11 if on else 8, fontweight="bold")

        # ---- nhãn xe ----
        r, c, heading, _ = env.state_decoder(self._obs)
        cx, cy = self._cell_xy(r, c)
        ax.text(cx, cy, 0.78, env.HEADING_ARROWS[heading],
                color="white", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

        ax.set_xlim(0, G); ax.set_ylim(0, G); ax.set_zlim(0, G * 0.6)
        try:
            ax.set_box_aspect((G, G, G * 0.5))
        except Exception:
            pass
        ax.view_init(elev=self._elev, azim=self._azim)
        self._canvas.draw_idle()

    def _add_car_faces(self, faces, fcols, ecols):
        env = self.env
        r, c, heading, goal_id = env.state_decoder(self._obs)
        cx, cy = self._cell_xy(r, c)
        body_color = AGENT_COLORS.get(self._sel_agent.get(), C_CAR_BODY)

        # thân xe
        self._add_box(faces, fcols, ecols,
                      cx - 0.30, cy - 0.30, 0.05, 0.60, 0.60, 0.24,
                      body_color, "white")
        # cabin
        self._add_box(faces, fcols, ecols,
                      cx - 0.17, cy - 0.17, 0.29, 0.34, 0.34, 0.20,
                      C_CAR_CABIN, "white")
        # mũi xe (khối vàng chỉ hướng đi)
        dvec = {env.NORTH: (0, 1), env.EAST: (1, 0),
                env.SOUTH: (0, -1), env.WEST: (-1, 0)}[heading]
        ncx = cx + dvec[0] * 0.34
        ncy = cy + dvec[1] * 0.34
        self._add_box(faces, fcols, ecols,
                      ncx - 0.10, ncy - 0.10, 0.10, 0.20, 0.20, 0.16,
                      C_NOSE, "#B45309")

    # ------------------------------------------------------------------ #
    #  Cập nhật panel thông tin                                            #
    # ------------------------------------------------------------------ #

    def _update_info(self, status_text):
        env = self.env
        r, c, heading, goal_id = env.state_decoder(self._obs)
        gr, gc = env.GOALS[goal_id]
        dist = abs(r - gr) + abs(c - gc)

        self._info["agent"].set(self._sel_agent.get())
        self._info["pos"].set(f"({r}, {c})")
        self._info["heading"].set(
            f"{env.HEADING_ARROWS[heading]} {env.HEADING_NAMES[heading]}")
        self._info["goal"].set(f"G{goal_id} ({gr},{gc})")
        self._info["dist"].set(str(dist))
        self._info["action"].set(
            env.ACTION_NAMES[self._last_action] if self._last_action is not None else "–")
        self._info["reward"].set(f"{self._last_reward:+.0f}")
        self._info["total"].set(f"{self._ep_reward:+.1f}")
        self._info["steps"].set(str(env.steps))
        self._info["status"].set(status_text)

    def _update_stats(self):
        self._stat_ep.set(f"Episode: {self._ep_count}")
        self._stat_win.set(f"Thành công: {self._wins}")
        rate = (self._wins / self._ep_count * 100) if self._ep_count else 0
        self._stat_rate.set(f"Tỷ lệ: {rate:.0f}%")

    # ------------------------------------------------------------------ #
    def _on_close(self):
        self._stop_auto()
        self.destroy()


def main():
    app = CarViz3D()
    app.mainloop()


if __name__ == "__main__":
    main()
