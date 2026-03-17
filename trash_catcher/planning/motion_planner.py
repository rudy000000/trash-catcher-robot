import numpy as np
from pydrake.all import (
    MathematicalProgram,
    Solve,
)


def plan_motion(
    pos_start: np.ndarray,
    pos_goal: np.ndarray,
    max_speed: float = 1.5,
    dt: float = 0.05,
    n_steps: int = 40,
) -> dict:
    """
    台車の最適経路を計画する。

    Args:
        pos_start: 開始位置 [x, y] (メートル)
        pos_goal:  目標位置 [x, y] (メートル)
        max_speed: 最大速度 (m/s)
        dt:        タイムステップ (秒)
        n_steps:   ステップ数

    Returns:
        positions: 経路上の位置列 (n_steps+1, 2)
        velocities: 速度列 (n_steps, 2)
        total_time: 合計時間 (秒)
    """
    prog = MathematicalProgram()

    # 位置・速度を決定変数として定義
    pos = prog.NewContinuousVariables(n_steps + 1, 2, "pos")
    vel = prog.NewContinuousVariables(n_steps, 2, "vel")

    # 初期・終端条件
    for i in range(2):
        prog.AddConstraint(pos[0, i] == pos_start[i])
        prog.AddConstraint(pos[-1, i] == pos_goal[i])

    # 運動学的制約: pos[t+1] = pos[t] + vel[t] * dt
    for t in range(n_steps):
        for i in range(2):
            prog.AddConstraint(
                pos[t + 1, i] == pos[t, i] + vel[t, i] * dt
            )

    # 速度制限
    for t in range(n_steps):
        for i in range(2):
            prog.AddConstraint(vel[t, i] <= max_speed)
            prog.AddConstraint(vel[t, i] >= -max_speed)

    # コスト: 速度の二乗和を最小化（滑らかな動きを優先）
    for t in range(n_steps):
        prog.AddCost(vel[t, 0] ** 2 + vel[t, 1] ** 2)

    result = Solve(prog)

    if not result.is_success():
        raise RuntimeError("経路計画の最適化に失敗しました")

    positions = result.GetSolution(pos)
    velocities = result.GetSolution(vel)

    return {
        "positions": positions,
        "velocities": velocities,
        "total_time": n_steps * dt,
    }


if __name__ == "__main__":
    pos_start = np.array([0.0, 0.0])
    pos_goal = np.array([2.0, 1.5])

    print("経路計画中...")
    result = plan_motion(pos_start, pos_goal)

    print(f"合計時間: {result['total_time']:.2f} 秒")
    print(f"開始位置: {result['positions'][0]}")
    print(f"終了位置: {result['positions'][-1].round(3)}")
    print(f"最大速度: {np.abs(result['velocities']).max():.3f} m/s")