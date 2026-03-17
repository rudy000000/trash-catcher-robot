import sys
sys.path.append('/workspace/trash_catcher/perception')
sys.path.append('/workspace/trash_catcher/planning')

from trash_catcher_env import TrashCatcherEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    env = TrashCatcherEnv()

    # 環境が正しく実装されているか確認
    print("環境チェック中...")
    check_env(env)
    print("環境チェック OK!")

    # PPOで学習
    print("学習開始...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    model.learn(total_timesteps=1_000_000)
    model.save("/workspace/trash_catcher/sim/ppo_trash_catcher")
    print("モデル保存完了!")

    # 学習後の成功率を評価
    print("\n--- 評価 ---")
    obs, _ = env.reset()
    success = 0
    trials = 20

    for _ in range(trials):
        obs, _ = env.reset()
        for _ in range(10):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                if reward > 0:
                    success += 1
                break

    print(f"成功率: {success}/{trials} ({success/trials*100:.1f}%)")