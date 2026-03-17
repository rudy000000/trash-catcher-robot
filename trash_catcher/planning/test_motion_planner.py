import numpy as np
import pytest
from motion_planner import plan_motion


class TestPlanMotion:

    def test_reaches_goal(self):
        """目標位置に到達するか"""
        pos_start = np.array([0.0, 0.0])
        pos_goal = np.array([2.0, 1.5])
        result = plan_motion(pos_start, pos_goal)
        assert result["positions"][-1] == pytest.approx(pos_goal, abs=0.01)

    def test_starts_at_origin(self):
        """開始位置が正しいか"""
        pos_start = np.array([1.0, 0.5])
        pos_goal = np.array([3.0, 2.0])
        result = plan_motion(pos_start, pos_goal)
        assert result["positions"][0] == pytest.approx(pos_start, abs=0.01)

    def test_speed_limit(self):
        """最大速度制限を超えないか"""
        pos_start = np.array([0.0, 0.0])
        pos_goal = np.array([2.0, 2.0])
        max_speed = 1.5
        result = plan_motion(pos_start, pos_goal, max_speed=max_speed)
        assert np.abs(result["velocities"]).max() <= max_speed + 0.001

    def test_total_time(self):
        """合計時間が正しいか"""
        result = plan_motion(
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            dt=0.05,
            n_steps=40
        )
        assert result["total_time"] == pytest.approx(2.0, abs=0.001)

    def test_same_start_and_goal(self):
        """開始と目標が同じ場合、速度がほぼ0か"""
        pos = np.array([1.0, 1.0])
        result = plan_motion(pos, pos)
        assert np.abs(result["velocities"]).max() == pytest.approx(0.0, abs=0.01)