"""
test_env.py
-----------
Unit tests cho DirectionalCarEnv.

Kiểm thử:
  1. Test boundary – xe không ra ngoài grid
  2. Test invalid action – raise AssertionError
  3. Test terminal state – va chạm & stop goal
  4. Test seed – reset với seed cho kết quả tái lập
  5. Test transition – heading update đúng

Chạy:
    python -m pytest tests/ -v
    python tests/test_env.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import DirectionalCarEnv


class TestDirectionalCarEnv(unittest.TestCase):

    def setUp(self):
        """Tạo môi trường mới cho mỗi test."""
        self.env = DirectionalCarEnv(max_steps=200)

    # ------------------------------------------------------------------ #
    #  Test 1: Khởi tạo                                                    #
    # ------------------------------------------------------------------ #

    def test_init(self):
        """n_states và n_actions phải đúng."""
        self.assertEqual(self.env.n_states,  7 * 7 * 4 * 3)   # 588
        self.assertEqual(self.env.n_actions, 4)

    def test_reset_returns_valid_state(self):
        """reset() trả về encoded state hợp lệ."""
        obs, info = self.env.reset(seed=0)
        self.assertIsInstance(obs, int)
        self.assertGreaterEqual(obs, 0)
        self.assertLess(obs, self.env.n_states)
        self.assertIsInstance(info, dict)

    def test_state_not_obstacle(self):
        """Vị trí xuất phát không phải obstacle."""
        for seed in range(20):
            obs, _ = self.env.reset(seed=seed)
            r, c, _, _ = self.env.state_decoder(obs)
            self.assertNotIn((r, c), self.env.OBSTACLES,
                             f"seed={seed}: xuất phát tại obstacle ({r},{c})")

    def test_state_not_goal(self):
        """Vị trí xuất phát không phải goal."""
        goals = set(self.env.GOALS)
        for seed in range(30):
            obs, _ = self.env.reset(seed=seed)
            r, c, _, _ = self.env.state_decoder(obs)
            self.assertNotIn((r, c), goals,
                             f"seed={seed}: xuất phát tại goal ({r},{c})")

    # ------------------------------------------------------------------ #
    #  Test 2: Boundary – xe không ra ngoài grid                           #
    # ------------------------------------------------------------------ #

    def test_forward_wall_top(self):
        """Xe đang ở hàng 0 quay NORTH không ra ngoài grid."""
        self.env._rng = __import__("numpy").random.RandomState(0)
        self.env._state  = (0, 3, self.env.NORTH, 0)
        self.env._steps  = 0
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        self.assertTrue(info["collision"], "Phải là collision khi chạm tường")
        self.assertTrue(terminated,        "Episode phải kết thúc sau va chạm")

    def test_forward_wall_bottom(self):
        """Xe đang ở hàng 6 quay SOUTH không ra ngoài grid."""
        self.env._state = (6, 3, self.env.SOUTH, 1)
        self.env._steps = 0
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        self.assertTrue(info["collision"])
        self.assertTrue(terminated)

    def test_forward_wall_left(self):
        """Xe đang ở cột 0 quay WEST không ra ngoài grid."""
        self.env._state = (3, 0, self.env.WEST, 2)
        self.env._steps = 0
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        self.assertTrue(info["collision"])
        self.assertTrue(terminated)

    def test_forward_wall_right(self):
        """Xe đang ở cột 6 quay EAST không ra ngoài grid."""
        self.env._state = (3, 6, self.env.EAST, 0)
        self.env._steps = 0
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        self.assertTrue(info["collision"])
        self.assertTrue(terminated)

    # ------------------------------------------------------------------ #
    #  Test 3: Invalid action                                              #
    # ------------------------------------------------------------------ #

    def test_invalid_action_raises(self):
        """step() với action ngoài [0,3] phải raise AssertionError."""
        self.env.reset(seed=0)
        with self.assertRaises(AssertionError):
            self.env.step(4)
        with self.assertRaises(AssertionError):
            self.env.step(-1)

    def test_step_before_reset_raises(self):
        """step() khi chưa reset() phải raise AssertionError."""
        fresh_env = DirectionalCarEnv()
        with self.assertRaises(AssertionError):
            fresh_env.step(0)

    # ------------------------------------------------------------------ #
    #  Test 4: Terminal state                                              #
    # ------------------------------------------------------------------ #

    def test_stop_at_goal_terminal(self):
        """STOP đúng tại goal → terminated=True, reached_goal=True."""
        goal_r, goal_c = self.env.GOALS[0]
        self.env._state  = (goal_r, goal_c, self.env.EAST, 0)
        self.env._steps  = 0
        _, reward, terminated, _, info = self.env.step(self.env.STOP)
        self.assertTrue(terminated,            "STOP tại goal phải terminal")
        self.assertTrue(info["reached_goal"],  "reached_goal phải True")
        self.assertEqual(reward, self.env.R_GOAL, "Reward phải là +50")

    def test_stop_wrong_not_terminal(self):
        """STOP sai vị trí → không terminal, reward âm."""
        self.env._state = (2, 2, self.env.EAST, 0)
        self.env._steps = 0
        _, reward, terminated, _, info = self.env.step(self.env.STOP)
        self.assertFalse(terminated,              "Stop sai không phải terminal")
        self.assertFalse(info["reached_goal"])
        self.assertEqual(reward,
                         self.env.R_STEP + self.env.R_WRONG_STOP,
                         "Reward sai STOP không đúng")

    def test_obstacle_collision_terminal(self):
        """Đi vào obstacle → terminated=True, collision=True."""
        # obstacle tại (1,1) – xe ở (1,0) quay EAST tiến vào (1,1)
        self.env._state = (1, 0, self.env.EAST, 0)
        self.env._steps = 0
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        self.assertTrue(terminated)
        self.assertTrue(info["collision"])
        self.assertEqual(reward,
                         self.env.R_STEP + self.env.R_COLLISION,
                         "Reward collision không đúng")

    def test_truncated_max_steps(self):
        """Episode bị truncated sau max_steps."""
        env = DirectionalCarEnv(max_steps=5)
        env.reset(seed=0)
        truncated_seen = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(env.TURN_LEFT)
            if truncated:
                truncated_seen = True
                break
            if terminated:
                break
        self.assertTrue(truncated_seen, "Episode phải bị truncated sau max_steps")

    # ------------------------------------------------------------------ #
    #  Test 5: Seed reproducibility                                        #
    # ------------------------------------------------------------------ #

    def test_seed_reproducibility(self):
        """reset(seed=X) cho cùng kết quả mỗi lần."""
        obs1, _ = self.env.reset(seed=42)
        obs2, _ = self.env.reset(seed=42)
        self.assertEqual(obs1, obs2, "reset(seed=42) phải cho cùng state")

    def test_different_seeds_different_states(self):
        """Các seed khác nhau cho state khác nhau (ít nhất một trong nhiều)."""
        states = set()
        for seed in range(10):
            obs, _ = self.env.reset(seed=seed)
            states.add(obs)
        self.assertGreater(len(states), 1,
                           "Các seed khác nhau phải cho state khác nhau")

    # ------------------------------------------------------------------ #
    #  Test 6: Transition – heading update                                 #
    # ------------------------------------------------------------------ #

    def test_turn_left_north_to_west(self):
        """TURN_LEFT từ NORTH → WEST."""
        self.env._state = (3, 3, self.env.NORTH, 0)
        self.env._steps = 0
        obs, _, _, _, _ = self.env.step(self.env.TURN_LEFT)
        _, _, new_heading, _ = self.env.state_decoder(obs)
        self.assertEqual(new_heading, self.env.WEST)

    def test_turn_right_north_to_east(self):
        """TURN_RIGHT từ NORTH → EAST."""
        self.env._state = (3, 3, self.env.NORTH, 0)
        self.env._steps = 0
        obs, _, _, _, _ = self.env.step(self.env.TURN_RIGHT)
        _, _, new_heading, _ = self.env.state_decoder(obs)
        self.assertEqual(new_heading, self.env.EAST)

    def test_turn_left_wraps(self):
        """TURN_LEFT từ NORTH (0) → WEST (3), không phải -1."""
        self.env._state = (3, 3, self.env.NORTH, 0)
        self.env._steps = 0
        obs, _, _, _, _ = self.env.step(self.env.TURN_LEFT)
        _, _, h, _ = self.env.state_decoder(obs)
        self.assertGreaterEqual(h, 0)
        self.assertLess(h, 4)

    def test_turn_right_wraps(self):
        """TURN_RIGHT từ WEST (3) → NORTH (0)."""
        self.env._state = (3, 3, self.env.WEST, 0)
        self.env._steps = 0
        obs, _, _, _, _ = self.env.step(self.env.TURN_RIGHT)
        _, _, h, _ = self.env.state_decoder(obs)
        self.assertEqual(h, self.env.NORTH)

    def test_forward_east(self):
        """FORWARD với heading EAST tăng col."""
        self.env._state = (3, 3, self.env.EAST, 1)
        self.env._steps = 0
        obs, _, terminated, _, _ = self.env.step(self.env.FORWARD)
        if not terminated:
            r, c, _, _ = self.env.state_decoder(obs)
            self.assertEqual(r, 3)
            self.assertEqual(c, 4)

    def test_forward_south(self):
        """FORWARD với heading SOUTH tăng row."""
        self.env._state = (2, 3, self.env.SOUTH, 1)
        self.env._steps = 0
        obs, _, terminated, _, _ = self.env.step(self.env.FORWARD)
        if not terminated:
            r, c, _, _ = self.env.state_decoder(obs)
            self.assertEqual(r, 3)
            self.assertEqual(c, 3)

    # ------------------------------------------------------------------ #
    #  Test 7: Render                                                      #
    # ------------------------------------------------------------------ #

    def test_render_before_reset(self):
        """render() khi chưa reset không gây lỗi."""
        fresh = DirectionalCarEnv()
        s = fresh.render()
        self.assertIsInstance(s, str)

    def test_render_returns_string(self):
        """render() sau reset trả về string có nội dung."""
        self.env.reset(seed=0)
        s = self.env.render()
        self.assertIsInstance(s, str)
        self.assertIn("Bước", s)
        self.assertIn("Goal", s)


# ======================================================================= #
#  Run                                                                      #
# ======================================================================= #

if __name__ == "__main__":
    unittest.main(verbosity=2)
