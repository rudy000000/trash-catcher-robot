import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.append('/workspace/trash_catcher/perception')
from trajectory import predict_landing


class TrashCatcherEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([-10.0, -10.0, -10.0, -10.0, 0.0]),
            high=np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
            dtype=np.float32
        )

        self.catch_radius = 0.5   # キャッチ判定を広げる
        self.max_speed = 2.0      # 速度を上げる
        self.dt = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ロボットと着地点の距離を制限（最大2m以内）
        self.landing_pos = self.np_random.uniform(-1.5, 1.5, size=2).astype(np.float32)
        offset = self.np_random.uniform(-2.0, 2.0, size=2).astype(np.float32)
        self.robot_pos = np.clip(self.landing_pos + offset, -5.0, 5.0).astype(np.float32)

        # 十分な時間を与える
        dist = np.linalg.norm(self.robot_pos - self.landing_pos)
        self.time_left = dist / self.max_speed * 2.5  # 余裕を2.5倍

        return self._get_obs(), {}

    def step(self, action):
        velocity = np.clip(action, -1.0, 1.0) * self.max_speed

        prev_dist = np.linalg.norm(self.robot_pos - self.landing_pos)

        self.robot_pos = np.clip(
            self.robot_pos + velocity * self.dt,
            -10.0, 10.0
        ).astype(np.float32)
        self.time_left = max(0.0, self.time_left - self.dt)

        dist = np.linalg.norm(self.robot_pos - self.landing_pos)
        caught = dist < self.catch_radius

        if caught:
            reward = 100.0
            terminated = True
        elif self.time_left <= 0:
            reward = -20.0
            terminated = True
        else:
            progress = prev_dist - dist
            reward = progress * 10.0
            terminated = False

        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # 相対位置を観測として使う（よりシンプル）
        relative = self.landing_pos - self.robot_pos
        return np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            relative[0],
            relative[1],
            self.time_left
        ], dtype=np.float32)


if __name__ == "__main__":
    env = TrashCatcherEnv()
    obs, _ = env.reset()
    print(f"初期観測: {obs}")
    print(f"着地点までの相対距離: ({obs[2]:.2f}, {obs[3]:.2f})")
    print(f"残り時間: {obs[4]:.2f}秒")