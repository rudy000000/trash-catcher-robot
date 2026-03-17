import numpy as np
from pydrake.all import LeafSystem


def predict_landing(pos0: np.ndarray, vel0: np.ndarray, g: float = 9.81) -> dict:
    """投げたゴミの落下点と到達時刻を予測する。"""
    if pos0[2] < 0:
        raise RuntimeError("初期位置が地面より下です")

    a = 0.5 * g
    b = -vel0[2]
    c = -pos0[2]

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise RuntimeError("着地点が存在しません")

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    t_land = max(t1, t2)

    return {
        "landing_x": pos0[0] + vel0[0] * t_land,
        "landing_y": pos0[1] + vel0[1] * t_land,
        "time_of_flight": t_land
    }


class TrajectoryEKF(LeafSystem):
    """カメラ観測にEKFを適用してノイズを除去するシステム。"""

    def __init__(self, dt: float = 0.033, g: float = 9.81):
        LeafSystem.__init__(self)
        self.dt = dt
        self.g = g
        self.n = 6
        self.x_est = np.zeros(self.n)
        self.P = np.eye(self.n) * 1.0
        self.Q = np.eye(self.n) * 0.01
        self.R = np.eye(3) * 0.05

    def state_transition(self, x):
        dt = self.dt
        F = np.eye(self.n)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        x_next = F @ x
        x_next[2] -= 0.5 * self.g * dt**2
        x_next[5] -= self.g * dt
        return x_next, F

    def update(self, z_obs: np.ndarray):
        x_pred, F = self.state_transition(self.x_est)
        P_pred = F @ self.P @ F.T + self.Q
        H = np.zeros((3, self.n))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        self.x_est = x_pred + K @ (z_obs - H @ x_pred)
        self.P = (np.eye(self.n) - K @ H) @ P_pred
        return self.x_est.copy()


if __name__ == "__main__":
    pos0 = np.array([0.0, 0.0, 2.0])
    vel0 = np.array([1.5, 0.0, 3.0])
    result = predict_landing(pos0, vel0)
    print(f"着地点X: {result['landing_x']:.3f} m")
    print(f"着地点Y: {result['landing_y']:.3f} m")
    print(f"飛行時間: {result['time_of_flight']:.3f} 秒")

    print("\n--- EKFテスト ---")
    ekf = TrajectoryEKF(dt=0.033)
    ekf.x_est = np.array([0.0, 0.0, 2.0, 1.5, 0.0, 3.0])
    for i in range(5):
        noise = np.random.normal(0, 0.05, 3)
        z_obs = ekf.x_est[:3] + noise
        state = ekf.update(z_obs)
        print(f"ステップ{i+1}: pos=({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}) "
              f"vel=({state[3]:.3f}, {state[4]:.3f}, {state[5]:.3f})")