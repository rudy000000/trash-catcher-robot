import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.append('/workspace/trash_catcher/perception')
from trajectory import predict_landing


class MuJoCoTrashCatcherEnv(gym.Env):
    """
    MuJoCoを使ったゴミキャッチロボット環境。
    物理シミュレーターで台車の動きをシミュレート。
    """

    def __init__(
        self,
        catch_radius: float = 0.5,
        max_speed: float = 2.0,
    ):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(
            '/workspace/trash_catcher/mujoco/omni_robot.xml'
        )
        self.data = mujoco.MjData(self.model)

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

        self.catch_radius = catch_radius
        self.max_speed = max_speed
        self.dt = 0.1
        self.landing_pos = np.zeros(2)
        self.time_left = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Domain Randomization: 床の摩擦をランダム化
        floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.model.geom_friction[floor_geom_id][0] = self.np_random.uniform(0.3, 1.0)

        # Domain Randomization: ロボットの質量をランダム化
        robot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )
        self.model.body_mass[robot_body_id] = self.np_random.uniform(2.5, 4.5)

        # Domain Randomization: アクチュエータのゲインをランダム化
        self.model.actuator_gainprm[0][0] = self.np_random.uniform(80, 120)
        self.model.actuator_gainprm[1][0] = self.np_random.uniform(80, 120)

        # ロボットをランダムな位置に配置
        robot_x = self.np_random.uniform(-1.0, 1.0)
        robot_y = self.np_random.uniform(-1.0, 1.0)
        self.data.qpos[0] = robot_x
        self.data.qpos[1] = robot_y

        # ゴミの初期位置・速度をランダム化
        pos0 = np.array([
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(1.5, 3.0)
        ])
        vel0 = np.array([
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(1.0, 4.0)
        ])

        try:
            result = predict_landing(pos0, vel0)
            self.landing_pos = np.clip(
                np.array([result["landing_x"], result["landing_y"]], dtype=np.float32),
                -5.0, 5.0
            )
            dist = np.linalg.norm(
                np.array([robot_x, robot_y]) - self.landing_pos
            )
            self.time_left = dist / self.max_speed * 2.5
        except RuntimeError:
            self.landing_pos = np.zeros(2, dtype=np.float32)
            self.time_left = 3.0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # 移動前の距離を先に記録
        robot_pos_before = self.data.qpos[:2].copy()
        prev_dist = np.linalg.norm(robot_pos_before - self.landing_pos)

        # アクションを速度としてMuJoCoに適用
        velocity = np.clip(action, -1.0, 1.0) * self.max_speed
        self.data.ctrl[0] = velocity[0]
        self.data.ctrl[1] = velocity[1]

        # 物理シミュレーションを10ステップ進める
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.time_left = max(0.0, self.time_left - self.dt)

        # 移動後の距離を取得
        robot_pos = self.data.qpos[:2].copy()
        dist = np.linalg.norm(robot_pos - self.landing_pos)
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
        robot_pos = self.data.qpos[:2].copy()
        relative = self.landing_pos - robot_pos

        # 実機のセンサーノイズを模擬
        noise = np.random.normal(0, 0.02, 5).astype(np.float32)

        obs = np.array([
            robot_pos[0],
            robot_pos[1],
            relative[0],
            relative[1],
            self.time_left
        ], dtype=np.float32)

        return np.clip(obs + noise, -10.0, 10.0)


if __name__ == "__main__":
    env = MuJoCoTrashCatcherEnv()
    obs, _ = env.reset()
    print(f"初期観測: {obs}")
    print(f"着地点までの相対距離: ({obs[2]:.2f}, {obs[3]:.2f})")
    print(f"残り時間: {obs[4]:.2f}秒")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"ステップ{i+1}: 報酬={reward:.2f} 終了={terminated}")
        if terminated:
            break