import sys
sys.path.append('/workspace/trash_catcher/perception')

from mujoco_env import MuJoCoTrashCatcherEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

SAVE_DIR = "/workspace/trash_catcher/mujoco/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":
    print("MuJoCo学習を開始します...")

    env = make_vec_env(MuJoCoTrashCatcherEnv, n_envs=1)
    eval_env = make_vec_env(MuJoCoTrashCatcherEnv, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=SAVE_DIR,
        eval_freq=10_000,
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    try:
        model.learn(
            total_timesteps=500_000,
            callback=eval_callback,
        )
    except KeyboardInterrupt:
        print("\n学習を停止しました")

    # 評価
    print("\n--- 評価 ---")
    best_model = PPO.load(f"{SAVE_DIR}/best_model")
    eval_env2 = MuJoCoTrashCatcherEnv()
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

    print(f"成功率: {success}/{trials} ({success/trials*100:.1f}%)")