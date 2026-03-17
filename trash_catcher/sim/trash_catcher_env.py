import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.append('/workspace/trash_catcher/perception')
sys.path.append('/workspace/trash_catcher/planning')
from trajectory import predict_landing, TrajectoryEKF
from motion_planner import plan_motion


class TrashCatcherEnv(gym.Env):
    """
    ゴミキャッチロボットのシミュレーション環境。
    観測: [ロボットx, ロボットy, 着地点x, 着地点y, 残り時間]
    行動: [目標x, 目標y] (移動先)
    報酬: キャッチ成功=+100, 失敗=-10, 近づくほど小報酬
    """

    def __init__(self):
        super().__init__()

        # 行動空間: 移動先 [x, y] (-5m ~ 5m)
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0]),
            high=np.array([5.0, 5.0]),
            dtype=np.float32
        )

        # 観測空間: [robot_x, robot_y, land_x, land_y, time_left]
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -5.0, 0.0]),
            high=np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
            dtype=np.float32
        )

        self.robot_pos = np.zeros(2)
        self.landing_pos = np.zeros(2)
        self.time_left = 0.0
        self.catch_radius = 0.3  # キャッチ判定半径 (m)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # ロボットをランダムな位置に配置
        self.robot_pos = self.np_random.uniform(-2.0, 2.0, size=2).astype(np.float32)

        # ゴミをランダムに投げる（Domain Randomization）
        pos0 = np.array([
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(1.5, 3.0)   # 高さ1.5m〜3m
        ])
        vel0 = np.array([
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(1.0, 4.0)   # 上向き成分
        ])

        result = predict_landing(pos0, vel0)
        self.landing_pos = np.array([
            result["landing_x"],
            result["landing_y"]
        ], dtype=np.float32)
        self.time_left = float(result["time_of_flight"])

        return self._get_obs(), {}

    def step(self, action):
        # 行動: 目標位置へ移動
        target = np.clip(action, -5.0, 5.0)

        # 移動距離に応じて時間を消費
        dist = np.linalg.norm(target - self.robot_pos)
        max_speed = 1.5
        move_time = dist / max_speed
        self.time_left -= move_time
        self.robot_pos = target.astype(np.float32)

        # 報酬計算
        dist_to_landing = np.linalg.norm(self.robot_pos - self.landing_pos)
        caught = dist_to_landing < self.catch_radius and self.time_left >= 0

        if caught:
            reward = 100.0
            terminated = True
        elif self.time_left < 0:
            reward = -10.0 - dist_to_landing  # 遅刻ペナルティ
            terminated = True
        else:
            reward = -dist_to_landing * 0.1   # 近づくほど小報酬
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

    # ランダム行動で5ステップ試す
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"ステップ{i+1}: 報酬={reward:.2f} 終了={terminated}")
        if terminated:
            break