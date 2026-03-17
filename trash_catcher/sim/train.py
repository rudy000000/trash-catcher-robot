import sys
sys.path.append('/workspace/trash_catcher/perception')
sys.path.append('/workspace/trash_catcher/planning')

from trash_catcher_env import TrashCatcherEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np


if __name__ == "__main__":

    # 並列環境で学習を安定化
    env = make_vec_env(TrashCatcherEnv, n_envs=4)
    eval_env = make_vec_env(TrashCatcherEnv, n_envs=1)

    # EvalCallbackで最良モデルを保存
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="/workspace/trash_catcher/sim/",
        log_path="/workspace/trash_catcher/sim/",
        eval_freq=10_000,
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

    print("学習開始...")
    model.learn(total_timesteps=500_000, callback=eval_callback)

    # 最良モデルで評価
    print("\n--- 評価 ---")
    best_model = PPO.load("/workspace/trash_catcher/sim/best_model")
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

    print(f"成功率: {success}/{trials} ({success/trials*100:.1f}%)")