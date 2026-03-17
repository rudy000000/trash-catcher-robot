import numpy as np
import pytest
from trajectory import predict_landing, TrajectoryEKF


class TestPredictLanding:

    def test_flat_throw(self):
        """水平投げの着地点テスト"""
        pos0 = np.array([0.0, 0.0, 2.0])
        vel0 = np.array([1.0, 0.0, 0.0])
        result = predict_landing(pos0, vel0)
        assert result["time_of_flight"] == pytest.approx(0.639, abs=0.01)
        assert result["landing_x"] == pytest.approx(0.639, abs=0.01)
        assert result["landing_y"] == pytest.approx(0.0, abs=0.001)

    def test_y_direction(self):
        """Y方向への投げテスト"""
        pos0 = np.array([0.0, 0.0, 1.0])
        vel0 = np.array([0.0, 2.0, 0.0])
        result = predict_landing(pos0, vel0)
        assert result["landing_x"] == pytest.approx(0.0, abs=0.001)
        assert result["landing_y"] > 0.0

    def test_high_throw(self):
        """上向きに投げた場合のテスト"""
        pos0 = np.array([0.0, 0.0, 1.0])
        vel0 = np.array([1.0, 0.0, 5.0])
        result = predict_landing(pos0, vel0)
        result_low = predict_landing(pos0, np.array([1.0, 0.0, 0.0]))
        assert result["time_of_flight"] > result_low["time_of_flight"]

    def test_negative_z_raises(self):
        """地面より下からの投げはエラー"""
        pos0 = np.array([0.0, 0.0, -1.0])
        vel0 = np.array([1.0, 0.0, -5.0])
        with pytest.raises(RuntimeError):
            predict_landing(pos0, vel0)


class TestEKF:

    def test_ekf_reduces_noise(self):
        """EKFが初期の推定誤差を時間とともに減らすか"""
        np.random.seed(42)
        ekf = TrajectoryEKF(dt=0.033)

        # 真の初期状態
        true_state = np.array([0.0, 0.0, 2.0, 1.5, 0.0, 3.0])
        ekf.x_est = true_state.copy()

        # 最初の誤差（初期ノイズを大きく設定）
        ekf.x_est[:3] += np.array([0.5, 0.5, 0.5])
        initial_error = np.linalg.norm(ekf.x_est[:3] - true_state[:3])

        # 正確な観測を10回与えてEKFを収束させる
        for _ in range(10):
            ekf.update(true_state[:3])

        final_error = np.linalg.norm(ekf.x_est[:3] - true_state[:3])

        # 収束後の誤差が初期誤差より小さいか
        assert final_error < initial_error