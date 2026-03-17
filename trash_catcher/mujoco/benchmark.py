import sys
sys.path.append('/workspace/trash_catcher/perception')

import numpy as np
import json
import os
from stable_baselines3 import PPO
from mujoco_env import MuJoCoTrashCatcherEnv

SAVE_DIR = "/workspace/trash_catcher/mujoco/checkpoints"
RESULTS_FILE = "/workspace/trash_catcher/mujoco/benchmark_results.json"
TRIALS = 100


def evaluate(model, env_kwargs: dict) -> float:
    """指定した環境設定で成功率を計測する"""
    success = 0
    for _ in range(TRIALS):
        env = MuJoCoTrashCatcherEnv(**env_kwargs)
        obs, _ = env.reset()
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                if reward == 100.0:
                    success += 1
                break
    return success / TRIALS * 100


if __name__ == "__main__":
    print("ベンチマーク開始...")
    model = PPO.load(f"{SAVE_DIR}/best_model")
    results = {}

    # 1. 成功率 vs キャッチ半径
    print("\n[1/6] キャッチ半径の影響を計測中...")
    results["catch_radius"] = {}
    for radius in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
        rate = evaluate(model, {"catch_radius": radius})
        results["catch_radius"][str(radius)] = rate
        print(f"  半径 {radius}m: {rate:.1f}%")

    # 2. 成功率 vs 台車最大速度
    print("\n[2/6] 台車速度の影響を計測中...")
    results["max_speed"] = {}
    for speed in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        rate = evaluate(model, {"max_speed": speed})
        results["max_speed"][str(speed)] = rate
        print(f"  速度 {speed}m/s: {rate:.1f}%")

    # 3. 成功率 vs 処理遅延
    print("\n[3/6] 処理遅延の影響を計測中...")
    results["delay"] = {}
    for delay in [0, 1, 2, 3, 5, 8]:
        success = 0
        for _ in range(TRIALS):
            env = MuJoCoTrashCatcherEnv()
            obs, _ = env.reset()
            for _ in range(50):
                action, _ = model.predict(obs, deterministic=True)
                # 遅延を模擬（古い観測を使う）
                for _ in range(delay):
                    obs_delayed = obs.copy()
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated:
                    if reward == 100.0:
                        success += 1
                    break
        rate = success / TRIALS * 100
        results["delay"][str(delay)] = rate
        print(f"  遅延 {delay}ステップ: {rate:.1f}%")

    # 4. 成功率 vs ノイズ強度
    print("\n[4/6] センサーノイズの影響を計測中...")
    results["noise"] = {}
    for noise_std in [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]:
        success = 0
        for _ in range(TRIALS):
            env = MuJoCoTrashCatcherEnv()
            obs, _ = env.reset()
            for _ in range(50):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                obs = obs + np.random.normal(0, noise_std, obs.shape).astype(np.float32)
                obs = np.clip(obs, -10.0, 10.0)
                if terminated:
                    if reward == 100.0:
                        success += 1
                    break
        rate = success / TRIALS * 100
        results["noise"][str(noise_std)] = rate
        print(f"  ノイズ σ={noise_std}: {rate:.1f}%")

    # 5. 成功率 vs ゴミの速度範囲
    print("\n[5/6] ゴミの速度範囲の影響を計測中...")
    results["trash_speed"] = {}
    for max_vel in [1.0, 2.0, 3.0, 4.0, 5.0]:
        success = 0
        for _ in range(TRIALS):
            env = MuJoCoTrashCatcherEnv()
            obs, _ = env.reset()
            env.landing_pos = np.clip(
                np.array([
                    np.random.uniform(-max_vel, max_vel),
                    np.random.uniform(-max_vel, max_vel)
                ], dtype=np.float32),
                -5.0, 5.0
            )
            for _ in range(50):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated:
                    if reward == 100.0:
                        success += 1
                    break
        rate = success / TRIALS * 100
        results["trash_speed"][str(max_vel)] = rate
        print(f"  最大速度 {max_vel}m/s: {rate:.1f}%")

    # 6. 成功率 vs ロボットと着地点の距離
    print("\n[6/6] 初期距離の影響を計測中...")
    results["initial_distance"] = {}
    for dist in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        success = 0
        for _ in range(TRIALS):
            env = MuJoCoTrashCatcherEnv()
            obs, _ = env.reset()
            angle = np.random.uniform(0, 2 * np.pi)
            env.landing_pos = np.array([
                env.data.qpos[0] + dist * np.cos(angle),
                env.data.qpos[1] + dist * np.sin(angle)
            ], dtype=np.float32)
            env.time_left = dist / env.max_speed * 2.5
            obs = env._get_obs()
            for _ in range(50):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated:
                    if reward == 100.0:
                        success += 1
                    break
        rate = success / TRIALS * 100
        results["initial_distance"][str(dist)] = rate
        print(f"  距離 {dist}m: {rate:.1f}%")

    # 結果を保存
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n結果を保存しました: {RESULTS_FILE}")