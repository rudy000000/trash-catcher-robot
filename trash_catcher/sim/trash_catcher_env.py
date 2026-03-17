import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.append('/workspace/trash_catcher/perception')
sys.path.append('/workspace/trash_catcher/planning')
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
            low=np.array([-5.0, -5.0, -5.0, -5.0, 0.0]),
            high=np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
            dtype=np.float32
        )

        self.robot_pos = np.zeros(2)
        self.landing_pos = np.zeros(2)
        self.time_left = 0.0
        self.catch_radius = 0.3
        self.max_speed = 1.5
        self.dt = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ロボットをランダムな位置に配置
        self.robot_pos = self.np_random.uniform(-2.0, 2.0, size=2).astype(np.float32)

        # Domain Randomization: ゴミの投げ方をランダム化
        pos0 = np.array([
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(1.0, 4.0)
        ])
        vel0 = np.array([
            self.np_random.uniform(-3.0, 3.0),
            self.np_random.uniform(-3.0, 3.0),
            self.np_random.uniform(0.5, 5.0)
        ])

        try:
            result = predict_landing(pos0, vel0)
            self.landing_pos = np.clip(
                np.array([result["landing_x"], result["landing_y"]], dtype=np.float32),
                -5.0, 5.0
            )
            self.time_left = float(result["time_of_flight"])
        except RuntimeError:
            self.landing_pos = np.zeros(2, dtype=np.float32)
            self.time_left = 2.0

        return self._get_obs(), {}

    def step(self, action):
        velocity = np.clip(action, -1.0, 1.0) * self.max_speed
        dt = self.dt

        prev_dist = np.linalg.norm(self.robot_pos - self.landing_pos)

        self.robot_pos = np.clip(
            self.robot_pos + velocity * dt,
            -5.0, 5.0
        ).astype(np.float32)
        self.time_left -= dt

        dist_to_landing = np.linalg.norm(self.robot_pos - self.landing_pos)
        caught = dist_to_landing < self.catch_radius and self.time_left >= 0

        if caught:
            reward = 100.0
            terminated = True
        elif self.time_left < 0:
            reward = -20.0
            terminated = True
        else:
            progress = prev_dist - dist_to_landing
            reward = progress * 5.0
            terminated = False

        truncated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.landing_pos[0],
            self.landing_pos[1],
            max(0.0, self.time_left)
        ], dtype=np.float32)


if __name__ == "__main__":
    env = TrashCatcherEnv()
    obs, _ = env.reset()
    print(f"初期観測: {obs}")
    print(f"着地点: ({obs[2]:.2f}, {obs[3]:.2f})")
    print(f"残り時間: {obs[4]:.2f}秒")