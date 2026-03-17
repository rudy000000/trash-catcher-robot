import numpy as np
import pytest
import sys
sys.path.append('/workspace/trash_catcher/perception')
from mujoco_env import MuJoCoTrashCatcherEnv


class TestMuJoCoTrashCatcherEnv:

    def setup_method(self):
        self.env = MuJoCoTrashCatcherEnv()

    def test_reset_returns_valid_obs(self):
        """resetが正しい形の観測を返すか"""
        obs, _ = self.env.reset(seed=42)
        assert obs.shape == (5,)
        assert self.env.observation_space.contains(obs)

    def test_action_space(self):
        """行動空間が正しいか"""
        assert self.env.action_space.shape == (2,)

    def test_catch_gives_positive_reward(self):
        """着地点に到達したら報酬100か"""
        self.env.reset(seed=42)
        self.env.landing_pos = self.env.data.qpos[:2].astype(np.float32)
        self.env.time_left = 5.0
        _, reward, terminated, _, _ = self.env.step(np.array([0.0, 0.0]))
        assert reward == 100.0
        assert terminated is True

    def test_timeout_gives_negative_reward(self):
        """時間切れで負の報酬か"""
        self.env.reset(seed=42)
        self.env.landing_pos = np.array([100.0, 100.0], dtype=np.float32)
        self.env.time_left = 0.05
        _, reward, terminated, _, _ = self.env.step(np.array([0.0, 0.0]))
        assert reward < 0
        assert terminated is True

    def test_obs_within_bounds(self):
        """観測値が常に境界内か"""
        self.env.reset(seed=42)
        for _ in range(10):
            action = self.env.action_space.sample()
            obs, _, terminated, _, _ = self.env.step(action)
            assert self.env.observation_space.contains(obs)
            if terminated:
                self.env.reset(seed=42)