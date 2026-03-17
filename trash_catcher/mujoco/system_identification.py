import numpy as np
import json
import sys
sys.path.append('/workspace/trash_catcher/perception')


class SystemIdentification:
    """
    実機のパラメータを測定してMuJoCoシミュレーターに反映するクラス。

    手順:
    1. 実機で測定実験を行う
    2. 測定データをこのクラスに入力する
    3. シミュレーターのパラメータを自動調整する
    """

    def __init__(self):
        self.params = {
            "friction": 0.7,          # 床の摩擦係数
            "robot_mass": 3.0,        # ロボットの質量 (kg)
            "actuator_gain": 100.0,   # アクチュエータのゲイン
            "max_speed": 2.0,         # 最大速度 (m/s)
            "motor_delay": 0.05,      # モーターの遅延 (秒)
        }

    def measure_friction(
        self,
        push_force: float,
        robot_mass: float,
        acceleration: float
    ) -> float:
        """
        摩擦係数を測定する。

        実験手順:
        1. ロボットを一定の力で押す
        2. 加速度を計測する
        3. F = ma + friction * m * g から摩擦係数を計算する

        Args:
            push_force: 押した力 (N)
            robot_mass: ロボットの質量 (kg)
            acceleration: 計測した加速度 (m/s^2)
        """
        g = 9.81
        friction = (push_force - robot_mass * acceleration) / (robot_mass * g)
        friction = max(0.1, min(2.0, friction))
        self.params["friction"] = friction
        print(f"摩擦係数: {friction:.3f}")
        return friction

    def measure_max_speed(
        self,
        distances: list,
        times: list
    ) -> float:
        """
        最大速度を測定する。

        実験手順:
        1. ロボットをフル出力で走らせる
        2. 複数の距離・時間を計測する
        3. 平均速度から最大速度を推定する

        Args:
            distances: 計測した距離のリスト (m)
            times: 計測した時間のリスト (秒)
        """
        speeds = [d / t for d, t in zip(distances, times)]
        max_speed = np.mean(speeds)
        self.params["max_speed"] = max_speed
        print(f"最大速度: {max_speed:.3f} m/s")
        return max_speed

    def measure_motor_delay(
        self,
        command_times: list,
        response_times: list
    ) -> float:
        """
        モーターの遅延を測定する。

        実験手順:
        1. コマンドを送った時刻を記録する
        2. ロボットが動き始めた時刻を記録する
        3. 差分の平均を遅延とする

        Args:
            command_times: コマンド送信時刻のリスト (秒)
            response_times: 動き始めた時刻のリスト (秒)
        """
        delays = [r - c for r, c in zip(response_times, command_times)]
        motor_delay = np.mean(delays)
        self.params["motor_delay"] = motor_delay
        print(f"モーター遅延: {motor_delay*1000:.1f} ms")
        return motor_delay

    def measure_robot_mass(
        self,
        weight_kg: float
    ) -> float:
        """
        ロボットの質量を設定する（体重計で直接計測）。

        Args:
            weight_kg: 計測した質量 (kg)
        """
        self.params["robot_mass"] = weight_kg
        print(f"ロボット質量: {weight_kg:.2f} kg")
        return weight_kg

    def save_params(self, path: str = "/workspace/trash_catcher/mujoco/real_params.json"):
        """測定したパラメータをJSONに保存する"""
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)
        print(f"パラメータを保存しました: {path}")

    def load_params(self, path: str = "/workspace/trash_catcher/mujoco/real_params.json"):
        """保存したパラメータを読み込む"""
        with open(path, "r") as f:
            self.params = json.load(f)
        print(f"パラメータを読み込みました: {self.params}")
        return self.params

    def apply_to_mujoco(self, model):
        """
        測定したパラメータをMuJoCoモデルに反映する。

        Args:
            model: MuJoCoのMjModelオブジェクト
        """
        import mujoco

        # 床の摩擦を更新
        floor_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        model.geom_friction[floor_geom_id][0] = self.params["friction"]

        # ロボットの質量を更新
        robot_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )
        model.body_mass[robot_body_id] = self.params["robot_mass"]

        # アクチュエータのゲインを更新
        gain = self.params["actuator_gain"]
        model.actuator_gainprm[0][0] = gain
        model.actuator_gainprm[1][0] = gain

        print("MuJoCoモデルにパラメータを反映しました")
        return model


if __name__ == "__main__":
    sysid = SystemIdentification()

    # 実機測定のサンプル（実際の値に置き換える）
    print("=== System Identification テスト ===\n")

    # 摩擦係数の測定例
    friction = sysid.measure_friction(
        push_force=15.0,    # 15Nで押した
        robot_mass=3.0,     # 3kgのロボット
        acceleration=3.2    # 3.2 m/s^2の加速度
    )

    # 最大速度の測定例
    max_speed = sysid.measure_max_speed(
        distances=[1.0, 2.0, 3.0],
        times=[0.52, 1.01, 1.53]
    )

    # モーター遅延の測定例
    delay = sysid.measure_motor_delay(
        command_times=[0.0, 1.0, 2.0],
        response_times=[0.048, 1.051, 2.049]
    )

    # 質量の測定例
    mass = sysid.measure_robot_mass(weight_kg=3.2)

    # パラメータを保存
    sysid.save_params()

    print("\n=== 最終パラメータ ===")
    for k, v in sysid.params.items():
        print(f"  {k}: {v}")