# Đề tài 15 – Xe tự hành mini trong grid có hướng

## Mô tả

Agent điều khiển xe đến đích trong grid 7×7, có hướng quay và vật cản.  
Xe phải **quay trước khi tiến** (không thể đổi hướng và tiến trong cùng một action).

---

## Cấu trúc dự án

```
project/
├── envs/
│   ├── base_env.py          Abstract base class
│   └── custom_env.py        DirectionalCarEnv (7×7)
├── agents/
│   ├── random_agent.py      Baseline: chọn ngẫu nhiên
│   ├── heuristic_agent.py   Baseline: quay về hướng goal
│   ├── q_learning.py        Q-Learning (off-policy)
│   └── sarsa.py             SARSA (on-policy)
├── experiments/
│   ├── configs.yaml         Siêu tham số
│   ├── train.py             Huấn luyện
│   └── evaluate.py          Đánh giá 10 seed
├── visualization/
│   ├── render.py            Vẽ grid bằng matplotlib
│   └── plots.py             Learning curves, policy heatmap
├── dashboard/
│   └── app.py               Dashboard tương tác
├── tests/
│   ├── test_env.py          Unit tests môi trường
│   ├── test_encoder.py      Unit tests encoder/decoder
│   └── test_rewards.py      Unit tests reward
├── reports/figures/         Hình xuất ra
├── main.py                  Entry point chính
└── requirements.txt
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

**Yêu cầu:** Python ≥ 3.8, numpy, matplotlib, pyyaml

---

## Sử dụng nhanh

```bash
# Xem menu
python main.py

# Huấn luyện tất cả agents (10 seed × 4 agents)
python main.py train

# Huấn luyện một agent cụ thể
python main.py train --agent q_learning --seed 0

# Đánh giá và in bảng mean±std
python main.py evaluate

# Mở dashboard tương tác
python main.py demo

# Train nhanh 3000 ep rồi mở dashboard ngay
python main.py demo --train_first

# Demo nhanh trên terminal (không GUI)
python main.py quickrun

# Vẽ learning curves và lưu hình
python main.py plots

# Chạy unit tests
python main.py test
```

---

## MDP

| Thành phần | Giá trị |
|-----------|---------|
| **State** | `(row, col, heading, goal_id)` |
| **Không gian trạng thái** | 7×7×4×3 = **588 states** |
| **Action** | `forward`, `turn_left`, `turn_right`, `stop` |
| **Heading** | NORTH(0), EAST(1), SOUTH(2), WEST(3) |
| **Goal** | 3 goals cố định: G0(0,6), G1(3,6), G2(6,0) |
| **Obstacle** | 7 ô: (1,1),(1,4),(2,4),(3,2),(4,4),(5,1),(5,5) |

### Bảng reward

| Hành động | Điều kiện | Reward | Terminal |
|-----------|-----------|--------|----------|
| FORWARD | Gần goal hơn | -1 + 1 = **0** | ✗ |
| FORWARD | Xa goal hơn | -1 + (-1) = **-2** | ✗ |
| FORWARD | Cùng khoảng cách | **-1** | ✗ |
| FORWARD | Va chạm tường/obstacle | -1 + (-30) = **-31** | ✓ |
| TURN_LEFT/RIGHT | Bất kỳ | **-1** | ✗ |
| STOP | Đúng tại goal | **+50** | ✓ |
| STOP | Sai vị trí | -1 + (-10) = **-11** | ✗ |

### Bản đồ (7×7)

```
  0  1  2  3  4  5  6
0  .  .  .  .  .  . G0
1  .  X  .  .  X  .  .
2  .  .  .  .  X  .  .
3  .  .  X  .  .  . G1
4  .  .  .  .  X  .  .
5  .  X  .  .  .  X  .
6 G2  .  .  .  .  .  .
```

---

## Thuật toán

### Q-Learning (off-policy)
```
Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
```
- **α** (alpha): tốc độ học, mặc định 0.1  
- **γ** (gamma): hệ số chiết khấu, mặc định 0.95  
- **TD target** = r + γ·max Q(s',·) (hoặc r nếu terminal)  
- **TD error** = TD_target − Q(s,a)

### SARSA (on-policy)
```
Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]
```
Khác Q-Learning ở chỗ dùng **a'** thực tế (epsilon-greedy) thay vì max.

### Epsilon-greedy
- ε ban đầu: 1.0 (khám phá 100%)
- ε suy giảm: `ε ← max(ε_end, ε × ε_decay)` sau mỗi episode
- ε tối thiểu: 0.01
- Đánh giá: ε = 0 (greedy hoàn toàn)

---

## Kết quả mục tiêu

| Chỉ số | Target |
|--------|--------|
| Success rate | ≥ 85% |
| Collision rate | ≤ 10% |
| Avg steps | Thấp hơn Random rõ ràng |

---

## Demo

Dashboard (`python main.py demo`) có:
- Lưới 7×7 với xe có mũi tên hướng
- Nút **▶ Step** / **⏩ Auto** / **↺ Reset**
- Chọn agent: Random / Heuristic / Q-Learning / SARSA
- Chọn goal: G0 / G1 / G2
- Policy view (mũi tên tại mỗi ô)
- Learning curve (sau khi train)

---

## Unit Tests

```
tests/test_env.py      – 15 test cases: boundary, invalid action, terminal, seed, transition
tests/test_encoder.py  – 9 test cases: round-trip, range, collision, biên, giá trị cụ thể
tests/test_rewards.py  – 11 test cases: mọi trường hợp reward
```

Chạy: `python main.py test`

---

## Lưu ý kỹ thuật

- **Không dùng** Gymnasium, Stable-Baselines, RLlib, hay bất kỳ thư viện RL có sẵn.
- Mọi thứ tự viết từ đầu.
- Terminal state được xử lý đúng trong Q-update: `target = r` (không cộng γ·Q(s')).
- SARSA cần `next_action` trước khi update – xem `experiments/train.py`.
- Đánh giá luôn với **epsilon = 0** (greedy mode).
