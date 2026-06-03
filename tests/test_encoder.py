"""
test_encoder.py
---------------
Unit tests cho state_encoder() và state_decoder().

Kiểm thử:
  1. encode → decode cho mọi state hợp lệ (round-trip)
  2. Tất cả encoded state nằm trong [0, n_states)
  3. Không có collision (hai state khác nhau → encoded khác nhau)
  4. Biên: state (0,0,0,0) và (6,6,3,2)
  5. Giá trị cụ thể kiểm tra công thức

Chạy:
    python tests/test_encoder.py
    python -m pytest tests/test_encoder.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import DirectionalCarEnv


class TestStateEncoder(unittest.TestCase):

    def setUp(self):
        self.env = DirectionalCarEnv()
        self.G = self.env.GRID_SIZE      # 7
        self.H = 4                        # headings
        self.N = len(self.env.GOALS)      # 3

    # ------------------------------------------------------------------ #
    #  Test 1: Round-trip (encode → decode → encode)                      #
    # ------------------------------------------------------------------ #

    def test_roundtrip_all_states(self):
        """
        Với mọi (r, c, h, g), encode rồi decode phải trả về state gốc.
        Kiểm tra toàn bộ 588 state.
        """
        count = 0
        for r in range(self.G):
            for c in range(self.G):
                for h in range(self.H):
                    for g in range(self.N):
                        state   = (r, c, h, g)
                        encoded = self.env.state_encoder(state)
                        decoded = self.env.state_decoder(encoded)
                        self.assertEqual(
                            state, decoded,
                            f"Round-trip fail: {state} → {encoded} → {decoded}"
                        )
                        count += 1
        self.assertEqual(count, self.G * self.G * self.H * self.N)

    # ------------------------------------------------------------------ #
    #  Test 2: Tất cả encoded nằm trong [0, n_states)                    #
    # ------------------------------------------------------------------ #

    def test_encoded_range(self):
        """Tất cả encoded state phải trong [0, n_states)."""
        n = self.env.n_states
        for r in range(self.G):
            for c in range(self.G):
                for h in range(self.H):
                    for g in range(self.N):
                        enc = self.env.state_encoder((r, c, h, g))
                        self.assertGreaterEqual(enc, 0)
                        self.assertLess(enc, n,
                                        f"state ({r},{c},{h},{g}) → {enc} ≥ n_states={n}")

    # ------------------------------------------------------------------ #
    #  Test 3: Không collision                                             #
    # ------------------------------------------------------------------ #

    def test_no_collision(self):
        """Hai state khác nhau phải cho encoded khác nhau."""
        encoded_set = set()
        total = 0
        for r in range(self.G):
            for c in range(self.G):
                for h in range(self.H):
                    for g in range(self.N):
                        enc = self.env.state_encoder((r, c, h, g))
                        encoded_set.add(enc)
                        total += 1
        self.assertEqual(
            len(encoded_set), total,
            f"Collision: {total} states nhưng chỉ {len(encoded_set)} encoded khác nhau"
        )

    # ------------------------------------------------------------------ #
    #  Test 4: Biên                                                        #
    # ------------------------------------------------------------------ #

    def test_boundary_state_0000(self):
        """State (0,0,0,0) → encoded = 0."""
        enc = self.env.state_encoder((0, 0, 0, 0))
        self.assertEqual(enc, 0)

    def test_boundary_state_max(self):
        """State (G-1, G-1, 3, N-1) → encoded = n_states - 1."""
        G, H, N = self.G, self.H, self.N
        enc = self.env.state_encoder((G-1, G-1, H-1, N-1))
        self.assertEqual(enc, self.env.n_states - 1,
                         f"State max phải encode thành {self.env.n_states - 1}")

    # ------------------------------------------------------------------ #
    #  Test 5: Giá trị cụ thể                                             #
    # ------------------------------------------------------------------ #

    def test_specific_value_east_goal1(self):
        """
        Kiểm tra công thức trực tiếp:
            (1, 2, EAST=1, goal_id=1)
            idx = 1*(7*4*3) + 2*(4*3) + 1*3 + 1
                = 84 + 24 + 3 + 1 = 112
        """
        state = (1, 2, self.env.EAST, 1)
        enc   = self.env.state_encoder(state)
        expected = 1*(7*4*3) + 2*(4*3) + 1*3 + 1
        self.assertEqual(enc, expected,
                         f"Encode ({state}) → {enc} ≠ expected {expected}")

    def test_specific_decode(self):
        """Decode 112 → (1, 2, 1, 1)."""
        decoded = self.env.state_decoder(112)
        self.assertEqual(decoded, (1, 2, 1, 1))

    def test_heading_ordering(self):
        """
        State với heading khác nhau (cùng r,c,g) phải cho encoded liên tiếp
        cách nhau N_GOALS = 3.
        """
        N = self.N
        r, c, g = 2, 3, 0
        encs = [self.env.state_encoder((r, c, h, g)) for h in range(4)]
        for i in range(1, 4):
            self.assertEqual(encs[i] - encs[i-1], N,
                             f"Heading ordering fail: encs={encs}")

    def test_goal_ordering(self):
        """
        State với goal_id khác nhau (cùng r,c,h) phải cho encoded liên tiếp +1.
        """
        r, c, h = 2, 3, 0
        encs = [self.env.state_encoder((r, c, h, g)) for g in range(self.N)]
        for i in range(1, self.N):
            self.assertEqual(encs[i] - encs[i-1], 1,
                             f"Goal ordering fail: encs={encs}")


# ======================================================================= #
#  Run                                                                      #
# ======================================================================= #

if __name__ == "__main__":
    unittest.main(verbosity=2)
