import numpy as np
import cv2
import sys
sys.path.append('/workspace/trash_catcher/perception')
from trajectory import predict_landing, TrajectoryEKF


class GarbageDetector:
    """
    カメラ映像からゴミを検出し、弾道を予測するクラス。
    色検出（HSV）でゴミを検出し、EKFでノイズを除去する。
    """

    def __init__(self, camera_height: float = 2.0, focal_length: float = 500.0):
        self.camera_height = camera_height
        self.focal_length = focal_length
        self.ekf = TrajectoryEKF(dt=0.033)
        self.is_tracking = False

        # 検出する色の範囲（赤いゴミを想定）
        self.lower_color = np.array([0, 100, 100])
        self.upper_color = np.array([10, 255, 255])

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        フレームからゴミを検出する。
        Returns: 検出結果 or None
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)

        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # 最大の輪郭を選択
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 画像座標を3D座標に変換
        h, w = frame.shape[:2]
        x_3d = (cx - w / 2) / self.focal_length * self.camera_height
        y_3d = (cy - h / 2) / self.focal_length * self.camera_height
        z_3d = self.camera_height

        return {
            "pixel_x": cx,
            "pixel_y": cy,
            "position_3d": np.array([x_3d, y_3d, z_3d])
        }

    def update_trajectory(self, position_3d: np.ndarray) -> dict | None:
        """
        EKFで軌道を更新し、着地点を予測する。
        """
        if not self.is_tracking:
            self.ekf.x_est = np.array([
                position_3d[0], position_3d[1], position_3d[2],
                0.0, 0.0, 0.0
            ])
            self.is_tracking = True

        state = self.ekf.update(position_3d)

        pos = state[:3]
        vel = state[3:]

        if pos[2] <= 0:
            return None

        try:
            result = predict_landing(pos, vel)
            return {
                "landing_x": result["landing_x"],
                "landing_y": result["landing_y"],
                "time_of_flight": result["time_of_flight"],
                "velocity": vel
            }
        except RuntimeError:
            return None

    def reset(self):
        """追跡をリセットする"""
        self.is_tracking = False
        self.ekf = TrajectoryEKF(dt=0.033)


if __name__ == "__main__":
    # テスト用の合成フレームで動作確認
    detector = GarbageDetector()

    # 赤いボールを含む合成フレームを作成
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1)

    result = detector.detect(frame)
    if result:
        print(f"検出成功!")
        print(f"画像座標: ({result['pixel_x']}, {result['pixel_y']})")
        print(f"3D座標: {result['position_3d'].round(3)}")

        traj = detector.update_trajectory(result['position_3d'])
        if traj:
            print(f"着地点: ({traj['landing_x']:.3f}, {traj['landing_y']:.3f})")
            print(f"飛行時間: {traj['time_of_flight']:.3f}秒")
    else:
        print("検出失敗")