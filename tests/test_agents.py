"""
tests/test_agents.py
--------------------
Unit tests cho HeuristicAgent (và RandomAgent).

Chạy:
    python -m pytest tests/test_agents.py -v
    python -m pytest tests/test_agents.py::TestHeuristicAgent -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.custom_env import DirectionalCarEnv
from agents.heuristic_agent import HeuristicAgent


class TestHeuristicAgent(unittest.TestCase):

    def setUp(self):
        self.env = DirectionalCarEnv(max_steps=200)
        self.agent = HeuristicAgent(self.env)

    # ------------------------------------------------------------------ #
    #  Khởi tạo                                                            #
    # ------------------------------------------------------------------ #

    def test_init_stores_env(self):
        """HeuristicAgent phải lưu trữ env."""
        self.assertIs(self.agent.env, self.env)

    # ------------------------------------------------------------------ #
    #  STOP khi đứng đúng goal                                             #
    # ------------------------------------------------------------------ #

    def test_stop_at_goal_0(self):
        """Đứng tại G0(0,6) với goal_id=0 → STOP."""
        state = self.env.state_encoder((0, 6, self.env.NORTH, 0))
        self.assertEqual(self.agent.select_action(state), self.env.STOP)

    def test_stop_at_goal_1(self):
        """Đứng tại G1(3,6) với goal_id=1 → STOP."""
        state = self.env.state_encoder((3, 6, self.env.EAST, 1))
        self.assertEqual(self.agent.select_action(state), self.env.STOP)

    def test_stop_at_goal_2(self):
        """Đứng tại G2(6,0) với goal_id=2 → STOP."""
        state = self.env.state_encoder((6, 0, self.env.SOUTH, 2))
        self.assertEqual(self.agent.select_action(state), self.env.STOP)

    def test_no_stop_when_not_at_goal(self):
        """Không đứng tại goal → không chọn STOP."""
        state = self.env.state_encoder((3, 3, self.env.NORTH, 0))
        action = self.agent.select_action(state)
        self.assertNotEqual(action, self.env.STOP)

    # ------------------------------------------------------------------ #
    #  FORWARD khi đúng hướng                                              #
    # ------------------------------------------------------------------ #

    def test_forward_when_facing_south_toward_goal(self):
        """
        Xe tại (0,0), goal G2(6,0): cần đi SOUTH.
        Nếu heading=SOUTH → FORWARD.
        """
        state = self.env.state_encoder((0, 0, self.env.SOUTH, 2))
        self.assertEqual(self.agent.select_action(state), self.env.FORWARD)

    def test_forward_when_facing_east_toward_goal(self):
        """
        Xe tại (0,0), goal G0(0,6): cần đi EAST.
        Nếu heading=EAST → FORWARD.
        """
        state = self.env.state_encoder((0, 0, self.env.EAST, 0))
        self.assertEqual(self.agent.select_action(state), self.env.FORWARD)

    def test_forward_when_facing_north_toward_goal(self):
        """
        Xe tại (6,6), goal G0(0,6): cần đi NORTH.
        Nếu heading=NORTH → FORWARD.
        """
        state = self.env.state_encoder((6, 6, self.env.NORTH, 0))
        self.assertEqual(self.agent.select_action(state), self.env.FORWARD)

    def test_forward_when_facing_west_toward_goal(self):
        """
        Xe tại (6,6), goal G2(6,0): cần đi WEST.
        Nếu heading=WEST → FORWARD.
        """
        state = self.env.state_encoder((6, 6, self.env.WEST, 2))
        self.assertEqual(self.agent.select_action(state), self.env.FORWARD)

    # ------------------------------------------------------------------ #
    #  TURN_RIGHT / TURN_LEFT khi cần xoay                                 #
    # ------------------------------------------------------------------ #

    def test_turn_right_when_needed(self):
        """
        Xe tại (0,0) heading=NORTH, goal G0(0,6): desired=EAST.
        (EAST - NORTH) % 4 = 1 → TURN_RIGHT.
        """
        state = self.env.state_encoder((0, 0, self.env.NORTH, 0))
        self.assertEqual(self.agent.select_action(state), self.env.TURN_RIGHT)

    def test_turn_left_when_needed(self):
        """
        Xe tại (0,0) heading=SOUTH, goal G0(0,6): desired=EAST.
        (EAST - SOUTH) % 4 = 3 → TURN_LEFT.
        """
        state = self.env.state_encoder((0, 0, self.env.SOUTH, 0))
        self.assertEqual(self.agent.select_action(state), self.env.TURN_LEFT)

    def test_turn_right_north_needs_east(self):
        """
        Xe tại (3,0) heading=NORTH, goal G1(3,6): desired=EAST → TURN_RIGHT.
        """
        state = self.env.state_encoder((3, 0, self.env.NORTH, 1))
        self.assertEqual(self.agent.select_action(state), self.env.TURN_RIGHT)

    def test_turn_left_east_needs_north(self):
        """
        Xe tại (3,3) heading=EAST, goal G0(0,6):
        dr=-3, dc=3 → |dr|==|dc|, ưu tiên dr → desired=NORTH.
        (NORTH - EAST) % 4 = 3 → TURN_LEFT.
        """
        state = self.env.state_encoder((3, 3, self.env.EAST, 0))
        action = self.agent.select_action(state)
        self.assertIn(action, (self.env.TURN_LEFT, self.env.TURN_RIGHT, self.env.FORWARD))

    # ------------------------------------------------------------------ #
    #  Giá trị trả về hợp lệ                                               #
    # ------------------------------------------------------------------ #

    def test_action_always_valid(self):
        """Mọi state hợp lệ đều trả về action trong {0,1,2,3}."""
        for s in range(self.env.n_states):
            action = self.agent.select_action(s)
            self.assertIn(action, (0, 1, 2, 3),
                          f"State {s} trả về action không hợp lệ: {action}")

    # ------------------------------------------------------------------ #
    #  Chạy một episode hoàn chỉnh                                         #
    # ------------------------------------------------------------------ #

    def test_episode_terminates(self):
        """Agent chạy không vô hạn – episode kết thúc trong max_steps."""
        env = DirectionalCarEnv(max_steps=500)
        agent = HeuristicAgent(env)
        state, _ = env.reset(seed=7)
        done = False
        for _ in range(500):
            action = agent.select_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done = True
                break
        self.assertTrue(done, "Episode không kết thúc trong 500 bước")

    def test_success_rate_reasonable(self):
        """
        Chạy 50 episode – tỉ lệ đến đích phải > 30%
        (heuristic không tránh obstacle nên không đạt 100%).
        """
        env = DirectionalCarEnv(max_steps=300)
        agent = HeuristicAgent(env)
        successes = 0
        n_episodes = 50
        for seed in range(n_episodes):
            state, _ = env.reset(seed=seed)
            for _ in range(300):
                action = agent.select_action(state)
                state, _, terminated, truncated, info = env.step(action)
                if terminated:
                    if info.get("reached_goal"):
                        successes += 1
                    break
                if truncated:
                    break
        rate = successes / n_episodes
        self.assertGreater(rate, 0.30,
                           f"Success rate quá thấp: {rate:.0%} (kỳ vọng > 30%)")

    # ------------------------------------------------------------------ #
    #  Các phương thức no-op không gây lỗi                                  #
    # ------------------------------------------------------------------ #

    def test_update_does_not_raise(self):
        """update() không học, không được raise."""
        state, _ = self.env.reset(seed=0)
        self.agent.update(state, 0, -1, state, False)

    def test_decay_epsilon_does_not_raise(self):
        """decay_epsilon() không được raise."""
        self.agent.decay_epsilon()

    def test_reset_does_not_raise(self):
        """reset() không được raise."""
        self.agent.reset()

    def test_save_load_do_not_raise(self):
        """save() và load() không được raise."""
        self.agent.save("dummy_path.pkl")
        self.agent.load("dummy_path.pkl")

    def test_name_attribute(self):
        """name phải là 'heuristic'."""
        self.assertEqual(self.agent.name, "heuristic")


# ======================================================================= #
#  Run                                                                      #
# ======================================================================= #

if __name__ == "__main__":
    unittest.main(verbosity=2)
