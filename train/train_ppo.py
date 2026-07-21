import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

# add parent dir to path so we can import env
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.rocket_env import RocketEnv


class TrainingLogger(BaseCallback):
    """logs reward and apogee error per episode during training"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_apogees = []
        self.current_reward = 0.0

    def _on_step(self):
        self.current_reward += self.locals["rewards"][0]

        # episode just ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            apogee = self.training_env.get_attr("apogee_altitude")[0]
            self.episode_apogees.append(apogee)
            self.current_reward = 0.0

        return True


def plot_results(logger, target_apogee, save_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # reward curve
    ax1.plot(logger.episode_rewards, alpha=0.4, color="steelblue", label="raw")

    # smoothed reward
    if len(logger.episode_rewards) > 50:
        smoothed = np.convolve(
            logger.episode_rewards,
            np.ones(50) / 50,
            mode="valid"
        )
        ax1.plot(smoothed, color="steelblue", linewidth=2, label="smoothed (50ep)")

    ax1.set_title("training reward over episodes")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("total reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # apogee convergence
    ax2.plot(logger.episode_apogees, alpha=0.4, color="coral", label="apogee achieved")
    ax2.axhline(target_apogee, color="green", linewidth=2, linestyle="--", label=f"target {target_apogee}m")

    if len(logger.episode_apogees) > 50:
        smoothed_apogee = np.convolve(
            logger.episode_apogees,
            np.ones(50) / 50,
            mode="valid"
        )
        ax2.plot(smoothed_apogee, color="coral", linewidth=2, label="smoothed (50ep)")

    ax2.set_title("apogee achieved vs target")
    ax2.set_xlabel("episode")
    ax2.set_ylabel("altitude (m)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_results.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"plot saved → {path}")


def evaluate(model, env, n_episodes=20):
    """run trained agent for n episodes, print apogee error stats"""
    errors = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        error = abs(env.apogee_altitude - env.target_apogee)
        errors.append(error)
        print(f"  ep {ep+1:02d} | apogee: {env.apogee_altitude:.1f}m | error: {error:.1f}m")

    print(f"\n  mean error : {np.mean(errors):.1f}m")
    print(f"  best error : {np.min(errors):.1f}m")
    print(f"  worst error: {np.max(errors):.1f}m")


def main():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    env = RocketEnv()

    # sanity check — catches common gym api mistakes
    print("checking environment...")
    check_env(env, warn=True)
    print("environment ok\n")

    # ppo with mlp policy — 2 hidden layers of 64 units each
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    logger = TrainingLogger()

    print("starting training...\n")
    model.learn(
        total_timesteps=500_000,
        callback=logger,
        progress_bar=True
    )

    # save model
    model_path = "results/models/ppo_airbrake"
    model.save(model_path)
    print(f"\nmodel saved → {model_path}")

    # plot training curves
    plot_results(logger, env.target_apogee, "results/plots")

    # evaluate trained agent
    print("\nevaluating trained agent (20 episodes):")
    evaluate(model, env, n_episodes=20)


if __name__ == "__main__":
    main()