import sys
sys.path.append('/workspace/trash_catcher/perception')
sys.path.append('/workspace/trash_catcher/planning')

from trash_catcher_env import TrashCatcherEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import numpy as np
import os
import json
from datetime import datetime


SAVE_DIR = "/workspace/trash_catcher/sim/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = f"{SAVE_DIR}/training_log.json"
logs = []


class LogCallback(EvalCallback):
    def _on_step(self):
        result = super()._on_step()
        if self.best_mean_reward != -np.inf:
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timesteps": self.num_timesteps,
                "best_mean_reward": float(self.best_mean_reward),
            }
            logs.append(log_entry)
            with open(LOG_FILE, "w") as f:
                json.dump(logs, f, indent=2)
        return result


if __name__ == "__main__":
    print("夜間学習を開始します...")
    print(f"最良モデルの保存先: {SAVE_DIR}")
    print(f"ログファイル: {LOG_FILE}")
    print("Ctrl+C で安全に停止できます\n")

    env = make_vec_env(TrashCatcherEnv, n_envs=1)
    eval_env = make_vec_env(TrashCatcherEnv, n_envs=1)

    eval_callback = LogCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=SAVE_DIR,
        eval_freq=5_000,       # 5000ステップごとに評価
        n_eval_episodes=50,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
    )

    try:
        model.learn(
            total_timesteps=3_000_000,  # 最大500万ステップ
            callback=eval_callback,
        )
    except KeyboardInterrupt:
        print("\n学習を停止しました")

    # 最終評価
    print("\n--- 最終評価 ---")
    best_model = PPO.load(f"{SAVE_DIR}/best_model")
    eval_env2 = TrashCatcherEnv()
    success = 0
    trials = 100

    for _ in range(trials):
        obs, _ = eval_env2.reset()
        for _ in range(50):
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env2.step(action)
            if terminated:
                if reward == 100.0:
                    success += 1
                break

    print(f"最終成功率: {success}/{trials} ({success/trials*100:.1f}%)")
    with open(LOG_FILE, "r") as f:
        data = json.load(f)
    if data:
        best = max(data, key=lambda x: x["best_mean_reward"])
        print(f"学習中の最高評価報酬: {best['best_mean_reward']:.1f} ({best['timestamp']})")