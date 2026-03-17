import numpy as np
import cv2
import pytest
import sys
sys.path.append('/workspace/trash_catcher/perception')
from camera_detector import GarbageDetector


class TestGarbageDetector:

    def setup_method(self):
        self.detector = GarbageDetector()

    def test_detects_red_ball(self):
        """赤いボールを検出できるか"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1)
        result = self.detector.detect(frame)
        assert result is not None
        assert result["pixel_x"] == pytest.approx(320, abs=5)
        assert result["pixel_y"] == pytest.approx(240, abs=5)

    def test_no_detection_on_empty_frame(self):
        """空のフレームでは検出しないか"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame)
        assert result is None

    def test_3d_position_returned(self):
        """3D座標が返されるか"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1)
        result = self.detector.detect(frame)
        assert result is not None
        assert result["position_3d"].shape == (3,)

    def test_trajectory_prediction(self):
        """軌道予測が返されるか"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(frame, (320, 240), 30, (0, 0, 255), -1)
        result = self.detector.detect(frame)
        assert result is not None
        traj = self.detector.update_trajectory(result["position_3d"])
        assert traj is not None
        assert "landing_x" in traj
        assert "landing_y" in traj
        assert "time_of_flight" in traj

    def test_reset_clears_tracking(self):
        """resetで追跡がリセットされるか"""
        self.detector.is_tracking = True
        self.detector.reset()
        assert self.detector.is_tracking is False