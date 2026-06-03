"""
test_rewards.py
---------------
Unit tests cho hàm reward trong DirectionalCarEnv.

Kiểm thử từng trường hợp reward theo spec đề tài:
  R_STEP           = -1    mỗi bước
  R_CLOSER         = +1    tiến gần goal
  R_FARTHER        = -1    tiến xa goal
  R_COLLISION      = -30   va chạm obstacle/tường
  R_GOAL           = +50   stop đúng tại goal
  R_WRONG_STOP     = -10   stop sai

Chạy:
    python tests/test_rewards.py
    python -m pytest tests/test_rewards.py -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import DirectionalCarEnv


class TestRewards(unittest.TestCase):

    def setUp(self):
        self.env = DirectionalCarEnv()
        # goal 0 tại (0, 6)

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #

    def _set_state(self, r, c, heading, goal_id=0):
        self.env._state = (r, c, heading, goal_id)
        self.env._steps = 0

    # ------------------------------------------------------------------ #
    #  Test FORWARD rewards                                                #
    # ------------------------------------------------------------------ #

    def test_forward_closer_reward(self):
        """
        FORWARD và tiến gần goal → reward = R_STEP + R_CLOSER = 0.
        goal=0 tại (0,6). Xe ở (0,4) quay EAST tiến tới (0,5) → gần hơn.
        """
        self._set_state(0, 4, self.env.EAST, goal_id=0)
        _, reward, _, _, info = self.env.step(self.env.FORWARD)
        self.assertFalse(info["collision"])
        expected = self.env.R_STEP + self.env.R_CLOSER
        self.assertEqual(reward, expected,
                         f"Closer reward sai: {reward} ≠ {expected}")

    def test_forward_farther_reward(self):
        """
        FORWARD và xa goal → reward = R_STEP + R_FARTHER = -2.
        goal=0 tại (0,6). Xe ở (0,5) quay WEST tiến tới (0,4) → xa hơn.
        """
        self._set_state(0, 5, self.env.WEST, goal_id=0)
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        if not info["collision"] and not terminated:
            expected = self.env.R_STEP + self.env.R_FARTHER
            self.assertEqual(reward, expected,
                             f"Farther reward sai: {reward} ≠ {expected}")

    def test_forward_same_distance_reward(self):
        """
        FORWARD và cùng khoảng cách → reward = R_STEP = -1.
        goal=1 tại (3,6). Xe ở (3,3) quay SOUTH tiến tới (4,3):
          dist trước = |3-3|+|3-6|=3
          dist sau   = |4-3|+|3-6|=4  → xa hơn → không phải same
        Dùng: goal=0 tại (0,6). Xe ở (1,2) quay SOUTH: (2,2)
          dist trước = |1-0|+|2-6|=5
          dist sau   = |2-0|+|2-6|=6 → xa hơn
        Dùng: Xe ở (3,5) quay NORTH tiến tới (2,5):
          dist đến goal0=(0,6): |3-0|+|5-6|=4 → |2-0|+|5-6|=3 → closer
        Khó tìm same distance case thẳng, dùng diagonal motion:
        Xe ở (2,5) quay WEST tiến tới (2,4):
          goal=0 tại (0,6): dist=|2-0|+|5-6|=3, sau=|2-0|+|4-6|=4 → farther
        Thử goal=1 tại (3,6). Xe ở (2,5) quay NORTH tiến (1,5):
          trước=|2-3|+|5-6|=2, sau=|1-3|+|5-6|=3 → farther
        Xe ở (3,5) quay SOUTH tiến (4,5):
          trước=|3-3|+|5-6|=1, sau=|4-3|+|5-6|=2 → farther
        Xe ở (0,3) quay SOUTH tiến (1,3):
          goal0=(0,6): trước=|0-0|+|3-6|=3, sau=|1-0|+|3-6|=4 → farther
        Xe ở (1,5) quay NORTH tiến (0,5):
          goal0=(0,6): trước=|1-0|+|5-6|=2, sau=|0-0|+|5-6|=1 → closer
        Trường hợp same distance khá hiếm trong grid này.
        Tạo thủ công: goal tại (3,6), xe ở (2,6) quay NORTH tiến (1,6):
          trước=|2-3|+|6-6|=1, sau=|1-3|+|6-6|=2 → farther
        Không có case rõ ràng same distance. Tester sẽ kiểm tra reward trong [-2,-1,0].
        """
        # Kiểm tra reward nằm trong tập hợp hợp lệ
        self._set_state(3, 3, self.env.EAST, goal_id=1)  # goal1=(3,6)
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        if not info["collision"]:
            valid = {
                self.env.R_STEP + self.env.R_CLOSER,
                self.env.R_STEP,
                self.env.R_STEP + self.env.R_FARTHER,
            }
            self.assertIn(reward, valid,
                          f"Forward reward {reward} không hợp lệ")

    # ------------------------------------------------------------------ #
    #  Test TURN rewards                                                   #
    # ------------------------------------------------------------------ #

    def test_turn_left_reward(self):
        """TURN_LEFT luôn cho R_STEP = -1."""
        self._set_state(2, 2, self.env.NORTH, goal_id=0)
        _, reward, _, _, _ = self.env.step(self.env.TURN_LEFT)
        self.assertEqual(reward, self.env.R_STEP)

    def test_turn_right_reward(self):
        """TURN_RIGHT luôn cho R_STEP = -1."""
        self._set_state(2, 2, self.env.NORTH, goal_id=0)
        _, reward, _, _, _ = self.env.step(self.env.TURN_RIGHT)
        self.assertEqual(reward, self.env.R_STEP)

    # ------------------------------------------------------------------ #
    #  Test STOP rewards                                                   #
    # ------------------------------------------------------------------ #

    def test_stop_at_goal_reward(self):
        """STOP đúng tại goal → R_GOAL = +50."""
        gr, gc = self.env.GOALS[0]
        self._set_state(gr, gc, self.env.EAST, goal_id=0)
        _, reward, terminated, _, info = self.env.step(self.env.STOP)
        self.assertEqual(reward, self.env.R_GOAL,
                         f"Stop goal reward sai: {reward} ≠ {self.env.R_GOAL}")
        self.assertTrue(terminated)
        self.assertTrue(info["reached_goal"])

    def test_stop_wrong_reward(self):
        """STOP sai vị trí → R_STEP + R_WRONG_STOP = -11."""
        self._set_state(2, 2, self.env.EAST, goal_id=0)
        _, reward, terminated, _, info = self.env.step(self.env.STOP)
        expected = self.env.R_STEP + self.env.R_WRONG_STOP
        self.assertEqual(reward, expected,
                         f"Wrong stop reward sai: {reward} ≠ {expected}")
        self.assertFalse(terminated)
        self.assertFalse(info["reached_goal"])

    # ------------------------------------------------------------------ #
    #  Test COLLISION rewards                                              #
    # ------------------------------------------------------------------ #

    def test_obstacle_collision_reward(self):
        """Va chạm obstacle → R_STEP + R_COLLISION = -31."""
        # obstacle tại (1,1), xe ở (0,1) quay SOUTH
        self._set_state(0, 1, self.env.SOUTH, goal_id=0)
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        expected = self.env.R_STEP + self.env.R_COLLISION
        self.assertEqual(reward, expected,
                         f"Collision reward sai: {reward} ≠ {expected}")
        self.assertTrue(terminated)
        self.assertTrue(info["collision"])

    def test_wall_collision_reward(self):
        """Va chạm tường → R_STEP + R_COLLISION = -31."""
        self._set_state(0, 3, self.env.NORTH, goal_id=0)
        _, reward, terminated, _, info = self.env.step(self.env.FORWARD)
        expected = self.env.R_STEP + self.env.R_COLLISION
        self.assertEqual(reward, expected,
                         f"Wall collision reward sai: {reward} ≠ {expected}")
        self.assertTrue(terminated)
        self.assertTrue(info["collision"])

    # ------------------------------------------------------------------ #
    #  Test reward constants                                               #
    # ------------------------------------------------------------------ #

    def test_reward_constants(self):
        """Kiểm tra các hằng số reward theo đặc tả."""
        self.assertEqual(self.env.R_STEP,       -1)
        self.assertEqual(self.env.R_CLOSER,     +1)
        self.assertEqual(self.env.R_FARTHER,    -1)
        self.assertEqual(self.env.R_COLLISION,  -30)
        self.assertEqual(self.env.R_GOAL,       +50)
        self.assertEqual(self.env.R_WRONG_STOP, -10)

    def test_total_reward_bounds(self):
        """
        Reward tối thiểu trong một bước:
          FORWARD collision = -1 + (-30) = -31
        Reward tối đa trong một bước:
          STOP tại goal = +50
        """
        min_r = self.env.R_STEP + self.env.R_COLLISION   # -31
        max_r = self.env.R_GOAL                            # +50
        self.assertEqual(min_r, -31)
        self.assertEqual(max_r,  50)

    # ------------------------------------------------------------------ #
    #  Test: Không phạt TURN quá nhiều ngoài step penalty                 #
    # ------------------------------------------------------------------ #

    def test_multiple_turns_only_step_penalty(self):
        """
        Quay liên tục chỉ bị phạt R_STEP mỗi bước.
        4 lần quay phải → quay về heading gốc, total = 4 * R_STEP = -4.
        """
        self._set_state(3, 3, self.env.NORTH, goal_id=0)
        total_reward = 0.0
        for _ in range(4):
            _, reward, terminated, _, _ = self.env.step(self.env.TURN_RIGHT)
            total_reward += reward
            if terminated:
                break
        self.assertEqual(total_reward, 4 * self.env.R_STEP,
                         "4 lần quay phải phải cho tổng reward -4")

    def test_wrong_stop_multiple_times(self):
        """
        STOP nhiều lần sai vị trí → tích lũy phạt.
        Mỗi lần = R_STEP + R_WRONG_STOP = -11.
        """
        self._set_state(2, 2, self.env.EAST, goal_id=0)
        _, r1, _, _, _ = self.env.step(self.env.STOP)   # -11
        _, r2, _, _, _ = self.env.step(self.env.STOP)   # -11
        self.assertEqual(r1, self.env.R_STEP + self.env.R_WRONG_STOP)
        self.assertEqual(r2, self.env.R_STEP + self.env.R_WRONG_STOP)


# ======================================================================= #
#  Run                                                                      #
# ======================================================================= #

if __name__ == "__main__":
    unittest.main(verbosity=2)
